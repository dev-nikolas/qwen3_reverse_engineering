# Copyright 2026 Alibaba Group and the Qwen Team.
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
Módulo de Extrapolación de Contexto YaRN (Yet another RoPE extensioN) para Qwen3.

Fundamento de Ingeniería Inversa:
Qwen3-1.7B cuenta con una ventana de atención nativa de 32,768 tokens configurada con
frecuencia angular base theta = 1,000,000. Para extender el contexto hasta 131,072 tokens
sin reentrenamiento completo ni degradación de perplejidad, se utiliza YaRN (Peng et al., 2023).

YaRN divide las frecuencias de RoPE en tres bandas según la longitud de onda:
1. Altas frecuencias (longitud de onda corta < beta_fast): Sin interpolación (preserva precisión local).
2. Bajas frecuencias (longitud de onda larga > beta_slow): Interpolación lineal clásica.
3. Frecuencias intermedias: Rampa suave de interpolación (fórmula sigmoidal / cúbica).
Además, aplica un factor de escala de temperatura a la atención para compensar la dilución de la entropía.
"""

import math
import torch
import torch.nn as nn
from .rotary import rotate_half

class Qwen3YaRNEmbedding(nn.Module):
    """
    Rotary Position Embeddings con algoritmo YaRN para contextos ultra-largos (hasta 131k tokens).
    """
    def __init__(
        self,
        dim: int = 128,
        max_position_embeddings: int = 32768,
        base: float = 1000000.0,
        scale_factor: float = 4.0,  # 32k * 4 = 128k / 131k tokens
        original_max_position_embeddings: int = 32768,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale_factor = scale_factor
        self.original_max_pos = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.device = device

        # Factor de corrección de temperatura de atención YaRN:
        # gamma = mscale * ln(scale_factor) + 1.0
        self.mscale_factor = mscale * math.log(scale_factor) + 1.0 if scale_factor > 1.0 else 1.0

        # Precalcular las frecuencias base
        inv_freq = self._compute_yarn_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_yarn_inv_freq(self) -> torch.Tensor:
        """Calcula el vector inverso de frecuencias aplicando la rampa de modulación YaRN."""
        pos = torch.arange(0, self.dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (pos / self.dim))

        if self.scale_factor <= 1.0:
            return inv_freq

        # Longitud de onda: lambda = 2 * pi / inv_freq
        wavelength = 2.0 * math.pi / inv_freq

        # Bandas de interpolación
        # rampa = (lambda - lambda_low) / (lambda_high - lambda_low)
        lambda_low = self.original_max_pos / self.beta_fast
        lambda_high = self.original_max_pos / self.beta_slow

        # Factor de rampa entre 0.0 y 1.0
        ramp = (wavelength - lambda_low) / (lambda_high - lambda_low + 1e-8)
        ramp = torch.clamp(ramp, min=0.0, max=1.0)

        # Frecuencia interpolada
        inv_freq_interpolated = inv_freq / self.scale_factor
        
        # Combinación suave YaRN: (1 - r) * no_interpolado + r * interpolado
        yarn_inv_freq = (1.0 - ramp) * inv_freq + ramp * inv_freq_interpolated
        return yarn_inv_freq

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Genera tensores cos y sin adaptados por YaRN para la longitud solicitada.
        
        Returns:
            cos, sin: [seq_len, dim]
        """
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(x.device))
        
        # Duplicar frecuencias para emparejar dimensiones (rotate_half)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = (emb.cos() * self.mscale_factor).to(x.dtype)
        sin = (emb.sin() * self.mscale_factor).to(x.dtype)
        return cos, sin

    def apply_yarn_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica rotación YaRN directa sobre consultas Q y claves K.
        """
        seq_len = q.shape[2] if q.dim() == 4 else q.shape[1]
        cos, sin = self.forward(q, seq_len)
        
        # Ajustar dimensiones para broadcasting con [batch, num_heads, seq_len, head_dim]
        if q.dim() == 4:
            cos = cos.unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, head_dim]
            sin = sin.unsqueeze(0).unsqueeze(1)
        elif q.dim() == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


__all__ = ["Qwen3YaRNEmbedding"]
