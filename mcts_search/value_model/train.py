import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType

class LossLoggingCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.losses = []
        os.makedirs(output_dir, exist_ok=True)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append({
                "step": state.global_step,
                "epoch": state.epoch,
                "loss": logs["loss"]
            })

            # save to file each time a record is made
            with open(os.path.join(self.output_dir, "loss_history.json"), "w") as f:
                json.dump(self.losses, f, indent=2)
                
    def on_train_end(self, args, state, control, **kwargs):
        # save the final version at the end of the training.
        with open(os.path.join(self.output_dir, "loss_history.json"), "w") as f:
            json.dump(self.losses, f, indent=2)

class PromptValueDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=8000):
        with open(data_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item['prompt']
        value = item['value']
        encoded = self.tokenizer(
            text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(value, dtype=torch.bfloat16)  # use bfloat16 to match model dtype
        }

class ValueHead(nn.Module):
    def __init__(self, hidden_size, head_init_std=0.01): # head_init_std should not be too large
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=head_init_std)
        nn.init.zeros_(self.value_head.bias)
        self.value_head_act = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

    def forward(self, last_hidden):
        x = self.dropout(last_hidden)
        value = self.value_head(x)
        value = self.value_head_act(value)
        return value.squeeze(-1)

class LLMWithValueHead(nn.Module):
    def __init__(self, base_model, head_init_std=0.01):
        super().__init__()
        self.llm = base_model
        self.value_head = ValueHead(self.llm.config.hidden_size, head_init_std)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)

        # last_token_hidden = last_hidden[:, -1, :]  # (batch, hidden)
        # values = self.value_head(last_token_hidden)  # (batch,)
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).bool()
        masked_hidden = last_hidden.masked_fill(~mask, 0.0)  # set the hidden states of padding tokens to 0
        pooled_hidden = masked_hidden.sum(dim=1) / mask.sum(dim=1) # (batch, hidden)
        values = self.value_head(pooled_hidden)  # (batch,)

        loss = None
        if labels is not None:
            # to ensure labels are in the same dtype and device as values
            labels = labels.to(values.dtype).to(values.device)
            loss = nn.MSELoss()(values, labels)
        #print(f"value: {values}, labels: {labels}, loss: {loss}")
        return {"loss": loss, "values": values}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the base model"""
        if hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the base model"""
        if hasattr(self.llm, 'gradient_checkpointing_disable'):
            self.llm.gradient_checkpointing_disable()

    def save_value_head(self, path):
        torch.save(self.value_head.state_dict(), path)

    def load_value_head(self, path):
        self.value_head.load_state_dict(torch.load(path, map_location=self.value_head.value_head.weight.device))

def analyze_dataset_token_lengths(dataset, tokenizer):
    print("[INFO] Analyzing dataset token lengths...")
    
    token_lengths = []
    
    for i in range(len(dataset)):
        sample = dataset.samples[i]
        text = sample['prompt']

        if len(text) > 100000:
            print(f"[WARNING] Sample {i} text is too long, skipping...")
            continue

        tokens = tokenizer(text, return_tensors='pt')
        token_length = len(tokens['input_ids'][0])
        token_lengths.append(token_length)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} samples...")
    
    import numpy as np
    token_lengths = np.array(token_lengths)
    
    min_length = np.min(token_lengths)
    max_length = np.max(token_lengths)
    mean_length = np.mean(token_lengths)
    median_length = np.median(token_lengths)
    p90_length = np.percentile(token_lengths, 90)
    p95_length = np.percentile(token_lengths, 95)
    p99_length = np.percentile(token_lengths, 99)
    
    print(f"\n=== Token Length Statistics ===")
    print(f"  Total samples: {len(token_lengths)}")
    print(f"  Min length:    {min_length}")
    print(f"  Max length:    {max_length}")
    print(f"  Mean length:   {mean_length:.1f}")
    print(f"  Median length: {median_length:.1f}")
    print(f"  90th percentile: {p90_length:.1f}")
    print(f"  95th percentile: {p95_length:.1f}")
    print(f"  99th percentile: {p99_length:.1f}")
    
    suggested_max_length = int(p95_length)
    print(f"\n=== Recommendations ===")
    print(f"  Suggested max_length (95th percentile): {suggested_max_length}")
    print(f"  Samples that would be truncated at {suggested_max_length}: {np.sum(token_lengths > suggested_max_length)} ({100 * np.sum(token_lengths > suggested_max_length) / len(token_lengths):.1f}%)")
    
    return {
        'min': min_length,
        'max': max_length,
        'mean': mean_length,
        'median': median_length,
        'p90': p90_length,
        'p95': p95_length,
        'p99': p99_length,
        'suggested_max_length': suggested_max_length
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="./value_model_round1")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_length', type=int, default=8000)
    parser.add_argument('--lora', action='store_true', help="Enable LoRA fine-tuning")
    parser.add_argument('--gpu_id', type=int, default=1, help="GPU device ID to use")
    parser.add_argument('--merge_lora', action='store_true', help="Merge LoRA weights into base model after training")
    args = parser.parse_args()

    # set to use only the specified GPU
    # this may modified
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[INFO] Set pad_token to eos_token")
    dataset = PromptValueDataset(args.train_data, tokenizer, max_length=args.max_length)
    
    # # analyze the dataset token lengths distribution
    # token_stats = analyze_dataset_token_lengths(dataset, tokenizer)
    
    # # give warning if max_length is too large or too small
    # if args.max_length > token_stats['p99']:
    #     print(f"[WARNING] Your max_length ({args.max_length}) is much larger than 99th percentile ({token_stats['p99']:.0f})")
    #     print(f"[WARNING] Consider using --max_length {int(token_stats['suggested_max_length'])} for better efficiency")
    # elif args.max_length < token_stats['p90']:
    #     truncated_ratio = (dataset.samples and 
    #                       sum(1 for sample in dataset.samples 
    #                           if len(tokenizer(sample['prompt'])['input_ids']) > args.max_length) / len(dataset.samples) * 100)
    #     print(f"[WARNING] Your max_length ({args.max_length}) will truncate many samples")
    #     print(f"[WARNING] Estimated truncation rate: {truncated_ratio:.1f}%")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        #device_map="auto"
    ).to(device)

    if args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # for Qwen
        )
        base_model = get_peft_model(base_model, lora_config)
        
        # to eusure LoRA parameters are trainable
        for name, param in base_model.named_parameters():
            #print(f"[INFO] Parameter: {name}, requires_grad: {param.requires_grad}")
            if 'lora_' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # # print info about trainable parameters
        # trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        # total_params = sum(p.numel() for p in base_model.parameters())
        # print(f"[INFO] LoRA enabled - Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        # print(f"[INFO] Total params: {total_params:,}")

    model = LLMWithValueHead(base_model)
    
    # to ensure Value Head parameters are trainable
    for param in model.value_head.parameters():
        param.requires_grad = True
    
    # to ensure Value Head and base model use the same dtype
    if torch.cuda.is_available():
        model.value_head = model.value_head.to(device).to(torch.bfloat16)
        print(f"[INFO] Value head moved to {device} with dtype bfloat16")
    
    # # print final trainable parameters
    # total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"[INFO] Final model - Trainable params: {total_trainable:,} ({100 * total_trainable / total_params:.2f}%)")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_strategy="epoch",
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        weight_decay=0.01,
        optim="adamw_torch",
        logging_dir=os.path.join(args.output_dir, "logs"),  # 添加日志目录
        report_to=["none"],  # 禁用默认的日志记录器，使用自己的
        max_grad_norm=1.0,
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        mse = ((preds - labels) ** 2).mean()
        mae = abs(preds - labels).mean()
        return {"mse": mse, "mae": mae}

    class ValueTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            self.use_lora = kwargs.pop('use_lora', False)
            self.merge_lora_on_save = kwargs.pop('merge_lora_on_save', False)
            self.tokenizer = kwargs.pop('tokenizer', None)
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs, labels=labels)
            loss = outputs["loss"]

            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            labels = inputs.pop("labels")
            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
            return outputs["loss"].detach(), outputs["values"].detach(), labels.detach()

        def _save(self, output_dir: str = None, state_dict=None):
            """ Rewrite the save method to save the full model at the end of each epoch """
            super()._save(output_dir, state_dict)
            
            if output_dir is None:
                output_dir = self.args.output_dir
            
            epoch = self.state.epoch
            print(f"[INFO] Saving model at epoch {epoch}...")
            
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            if self.use_lora:
                print(f"[INFO] Saving LoRA adapter to {epoch_dir}")
                
                if self.merge_lora_on_save:
                    print("[INFO] Merging LoRA weights into base model...")
                    merged_model = self.model.llm.merge_and_unload()
                    
                    merged_model.save_pretrained(epoch_dir)
                    if self.tokenizer:
                        self.tokenizer.save_pretrained(epoch_dir)
                    print(f"[INFO] Merged model saved to {epoch_dir}")
                else:
                    # only save LoRA adapter weights
                    self.model.llm.save_pretrained(epoch_dir)
                    if self.tokenizer:
                        self.tokenizer.save_pretrained(epoch_dir)
            else:
                # full model saving
                self.model.llm.save_pretrained(epoch_dir)
                if self.tokenizer:
                    self.tokenizer.save_pretrained(epoch_dir)
                print(f"[INFO] Full model saved to {epoch_dir}")
            
            # save the value head
            value_head_path = os.path.join(epoch_dir, "value_head.pt")
            self.model.save_value_head(value_head_path)
            print(f"[INFO] Value head saved to {value_head_path}")
            
            print(f"[INFO] Epoch {epoch} model saved successfully to {epoch_dir}")
    
    loss_callback = LossLoggingCallback(args.output_dir)

    trainer = ValueTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        compute_metrics=compute_metrics,
        use_lora=args.lora,
        merge_lora_on_save=args.merge_lora,
        tokenizer=tokenizer,
        callbacks=[loss_callback]
    )

    trainer.train()

if __name__ == "__main__":
    main()
