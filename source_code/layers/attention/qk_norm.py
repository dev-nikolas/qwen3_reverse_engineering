# Copyright 2025 The Qwen team, Alibaba Group and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn
from ..normalization.rmsnorm import RMSNorm

class QKNorm(nn.Module):
    """
    Normalizador de Consultas y Claves (QK-Norm) para atención multi-cabeza.
    Aplica RMSNorm independiente por cabeza antes del producto punto escalado,
    previniendo que las activaciones de atención diverjan en secuencias largas.
    """
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)
