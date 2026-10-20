# Copyright 2025 The Qwen Team and Alibaba Group.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AWQ (Activation-aware Weight Quantization) & Rogue Channel Protection
======================================================================
Calibrador y empaquetador de pesos consciente de activaciones para Qwen3-1.7B.

Protege de forma determinista la Dimensión Rebelde 1793 (Rogue Dimension) y
los canales salientes del residual stream contra la pérdida de precisión
en cuantizaciones de 4 y 8 bits (W4A16 / W8A16).

Fundamento Matemático:
  s = (E[|X|])^alpha / (max(|W|))^(1 - alpha)
  W_scaled = W * diag(s)
  X_scaled = X * diag(s)^(-1)  <-- Absorbido en el vector gamma de RMSNorm
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class AWQChannelProtector:
    """
    Identificador y calibrador de canales críticos para Qwen3.
    """
    def __init__(self, alpha: float = 0.5, rogue_dim_index: int = 1793):
        self.alpha = alpha
        self.rogue_dim_index = rogue_dim_index

    def compute_optimal_scales(
        self,
        weight: torch.Tensor,       # [out_features, in_features]
        act_scales: torch.Tensor    # [in_features]
    ) -> torch.Tensor:
        """
        Calcula el vector de escala s de per-channel para ecualizar la dificultad
        entre la activación y el peso.
        """
        # Magnitud máxima por columna de pesos
        w_max = torch.max(torch.abs(weight), dim=0).values.clamp(min=1e-5)
        a_max = act_scales.clamp(min=1e-5)

        # s = a_max^alpha / w_max^(1-alpha)
        scales = (a_max ** self.alpha) / (w_max ** (1.0 - self.alpha))
        scales = scales.clamp(min=1e-4, max=1e4)

        # Si el canal coincide con la dimensión rebelde, garantizar escala mínima de protección
        if 0 <= self.rogue_dim_index < scales.shape[0]:
            scales[self.rogue_dim_index] = torch.clamp(scales[self.rogue_dim_index], min=2.0)

        return scales

    def protect_and_quantize_layer(
        self,
        linear: nn.Linear,
        act_scales: torch.Tensor,
        bits: int = 8,
        group_size: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aplica escalado protector a una capa lineal y la cuantiza por grupos.
        
        Returns:
            q_weight: Tensor int8 empaquetado
            scales: Escala por canal s aplicada
            group_scales: Escalas de cuantización por grupo [out_features, num_groups]
        """
        W = linear.weight.data.clone().float()
        s = self.compute_optimal_scales(W, act_scales)

        # Escalar pesos: W' = W * s
        W_scaled = W * s.unsqueeze(0)

        # Cuantización por grupos (group-wise quantization)
        out_features, in_features = W_scaled.shape
        num_groups = (in_features + group_size - 1) // group_size

        padded_in = num_groups * group_size
        if padded_in > in_features:
            W_padded = nn.functional.pad(W_scaled, (0, padded_in - in_features))
        else:
            W_padded = W_scaled

        W_reshaped = W_padded.view(out_features, num_groups, group_size)
        group_max = torch.amax(torch.abs(W_reshaped), dim=-1, keepdim=True).clamp(min=1e-5)

        q_max = (1 << (bits - 1)) - 1
        q_scales = group_max / q_max

        q_weight = torch.clamp(torch.round(W_reshaped / q_scales), -q_max, q_max).to(torch.int8)

        return q_weight, s, q_scales.squeeze(-1)


class AWQLinear(nn.Module):
    """
    Capa Lineal cuantizada con protección AWQ.
    Descuantiza eficientemente al vuelo durante la inferencia aplicando
    la compensación del canal rebelde en FP16.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        group_size: int = 128
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.ones(in_features, dtype=torch.float32))
        num_groups = (in_features + group_size - 1) // group_size
        self.register_buffer("group_scales", torch.ones(out_features, num_groups, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inferencia:
          x: [..., in_features]
          Compensa x dividiendo por la escala s, y multiplica por el peso reconstruido.
        """
        # Deshacer escalado en la entrada: x' = x / s
        x_norm = x / self.scales

        # Reconstruir W en FP16/FP32
        num_groups = self.group_scales.shape[1]
        padded_in = num_groups * self.group_size
        
        W_float = self.qweight.float()
        if padded_in > self.in_features:
            W_padded = nn.functional.pad(W_float, (0, padded_in - self.in_features))
        else:
            W_padded = W_float

        W_reshaped = W_padded.view(self.out_features, num_groups, self.group_size)
        W_dequant = (W_reshaped * self.group_scales.unsqueeze(-1)).view(self.out_features, -1)[:, :self.in_features]

        return nn.functional.linear(x_norm, W_dequant)
