# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class ValueHead(nn.Module):
    def __init__(self, hidden_size, head_init_std=0.01):
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

class ValueModel:
    def __init__(self, model_name_or_path, value_head_path=None, device="cuda:1", dtype=torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        hidden_size = self.model.config.hidden_size
        self.value_head = ValueHead(hidden_size)
        self.value_head.to(device).to(dtype)
        if value_head_path is not None:
            self.value_head.load_state_dict(torch.load(value_head_path, map_location=device))
        self.value_head.eval()

        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, prompts, max_length=8000):
        # batch size should be small enough to fit in memory
        encoded = self.tokenizer(
            prompts, add_special_tokens=True,
            padding=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # right truncation to keep only the last max_length tokens
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )
        last_hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)
        return last_hidden, attention_mask

    @torch.no_grad()
    def score_mean_pooling(self, prompts, max_length=8000, batch_size=10, show_progress=True):
        results = []
        total = len(prompts)
        iterator = range(0, total, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Batch Scoring", ncols=90)
        for i in iterator:
            batch_prompts = prompts[i:i+batch_size]
            last_hidden, attention_mask = self.encode(batch_prompts, max_length=max_length)
            mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).bool()
            masked_hidden = last_hidden.masked_fill(~mask, 0.0)
            pooled_hidden = masked_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            values = self.value_head(pooled_hidden)
            results.extend(values.cpu().tolist())
        return results

    def save_value_head(self, path):
        torch.save(self.value_head.state_dict(), path)

    def load_value_head(self, path):
        self.value_head.load_state_dict(torch.load(path, map_location=self.device))

if __name__ == "__main__":
    model_dir = "../value_model/value_model_round1_7B_NEW/checkpoint-63309/epoch_3.0"
    value_head_pt = model_dir + "/value_head.pt"
    model = ValueModel(model_dir, value_head_pt)

    prompts = [
        "Only work but no play makes Jack a dull\n" * 10000 for _ in range(100)  # long prompts
    ]
    print("[main] scores:", model.score_mean_pooling(prompts, max_length=8000, batch_size=10, show_progress=True))
