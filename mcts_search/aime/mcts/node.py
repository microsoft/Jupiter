# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

class BaseNode:
    def __init__(self, parent=None):
        self.tag = "0"
        self.state = dict()
        self.parent = parent
        self.children = []
        self.depth = 0
        self.is_terminal = False
        self.consecutive_errors = 0
        # self.reward = 0
        self.value = -100

    def has_children(self):
        return len(self.children) > 0

    def is_root(self):
        return self.parent is None

class MCTSNode(BaseNode):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.prior = 1.0
        self.c_puct = 0

        self.visit_count = 0
        self.value_sum = 0.0

    def q_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def update_visit_count(self, count):
        self.visit_count = count

    def update(self, value):
        if self.value == -100:
            self.value = value
        self.visit_count += 1
        self.value_sum += value

    def update_recursive(self, value, start_node):
        self.update(value)
        if self.tag == start_node.tag:
            return
        self.parent.update_recursive(value, start_node)

    def puct(self) -> float:
        q_value = self.q_value() if self.visit_count > 0 else 0
        u_value = 0
        if self.parent is not None:
            u_value = self.c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q_value + u_value
