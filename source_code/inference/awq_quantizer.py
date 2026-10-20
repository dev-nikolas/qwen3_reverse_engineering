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

import torch  # type: ignore
import torch.nn as nn  # type: ignore
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


def pack_int4_weights(q_int8: torch.Tensor) -> torch.Tensor:
    """
    Empaqueta tensores int8 de 4 bits (-8 a +7) en enteros uint8 compactos.
    Almacena dos nibbles de 4 bits por cada byte, reduciendo el tamaño en memoria al 50%.
    """
    clamped = torch.clamp(q_int8, -8, 7).to(torch.int32)
    unsigned = clamped + 8  # Mapea de [-8, 7] a [0, 15]

    orig_last_dim = unsigned.shape[-1]
    if orig_last_dim % 2 != 0:
        unsigned = nn.functional.pad(unsigned, (0, 1), value=8)

    low = unsigned[..., 0::2]
    high = unsigned[..., 1::2]
    packed = (low | (high << 4)).to(torch.uint8)
    return packed


def unpack_int4_weights(packed_uint8: torch.Tensor, target_in_features: int) -> torch.Tensor:
    """
    Desempaqueta tensores uint8 compactos recuperando los valores originales en int8 con signo (-8 a +7).
    """
    low = (packed_uint8 & 0x0F).to(torch.int32) - 8
    high = ((packed_uint8 >> 4) & 0x0F).to(torch.int32) - 8

    # Intercalar low y high en la última dimensión
    unpacked = torch.stack([low, high], dim=-1).flatten(-2, -1)
    if unpacked.shape[-1] > target_in_features:
        unpacked = unpacked[..., :target_in_features]
    return unpacked.to(torch.int8)


class AWQ4BitLinear(nn.Module):
    """
    Capa Lineal ultra-comprimida en 4 bits (W4A16) con protección de canales sobresalientes.
    
    Características:
      - Almacena pesos comprimidos en uint8 (2 nibbles/byte), requiriendo 75% menos VRAM que FP16.
      - Descuantización por grupos (group_size=128 por defecto).
      - Protección determinista de la Dimensión Rebelde 1793 mediante vector de escala 'scales'.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        packed_cols = (in_features + 1) // 2
        self.register_buffer("packed_weight", torch.zeros(out_features, packed_cols, dtype=torch.uint8))
        self.register_buffer("scales", torch.ones(in_features, dtype=torch.float32))

        num_groups = (in_features + group_size - 1) // group_size
        self.register_buffer("group_scales", torch.ones(out_features, num_groups, dtype=torch.float32))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        act_scales: torch.Tensor,
        group_size: int = 128,
        protector: Optional[AWQChannelProtector] = None
    ) -> "AWQ4BitLinear":
        """
        Construye una capa AWQ4BitLinear a partir de una capa nn.Linear calibrada.
        """
        if protector is None:
            protector = AWQChannelProtector(alpha=0.5, rogue_dim_index=1793)

        out_features, in_features = linear.weight.shape
        q_weight_8, s, group_scales = protector.protect_and_quantize_layer(
            linear=linear,
            act_scales=act_scales,
            bits=4,
            group_size=group_size
        )

        packed = pack_int4_weights(q_weight_8.view(out_features, -1)[:, :in_features])

        layer = cls(in_features=in_features, out_features=out_features, group_size=group_size)
        layer.packed_weight.copy_(packed)
        layer.scales.copy_(s)
        layer.group_scales.copy_(group_scales)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inferencia W4A16:
          1. Desescala la entrada con s (protegiendo el canal 1793).
          2. Desempaqueta y descuantiza los pesos W a FP32/FP16.
          3. Ejecuta la proyección lineal.
        """
        # Deshacer escalado en la entrada: x' = x / s
        x_norm = x / self.scales

        # Desempaquetar pesos de 4 bits
        unpacked_int8 = unpack_int4_weights(self.packed_weight, self.in_features).float()

        # Descuantizar por grupos
        num_groups = self.group_scales.shape[1]
        padded_in = num_groups * self.group_size
        if padded_in > self.in_features:
            W_padded = nn.functional.pad(unpacked_int8, (0, padded_in - self.in_features))
        else:
            W_padded = unpacked_int8

        W_reshaped = W_padded.view(self.out_features, num_groups, self.group_size)
        W_dequant = (W_reshaped * self.group_scales.unsqueeze(-1)).view(self.out_features, -1)[:, :self.in_features]

        return nn.functional.linear(x_norm, W_dequant)

