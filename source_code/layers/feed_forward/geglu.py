# Copyright 2025 The Qwen team, Alibaba Group and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

class GeGLU(nn.Module):
    """
    Red Feed-Forward con activación Gaussian Error Gated Linear Unit (GeGLU).
    Utilizada en arquitecturas como Gemma 2 para comparativa.
    Fórmula: down_proj( GELU(gate_proj(x)) * up_proj(x) )
    """
    def __init__(self, hidden_size: int = 2048, intermediate_size: int = 6144):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))
