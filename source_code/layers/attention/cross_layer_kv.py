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
Cross-Layer Attention & KV-Sharing Engine (CLA)
===============================================
Mecanismo de compartición de Claves y Valores entre capas contiguas (Capas Pares e Impares)
para reducir el tráfico de memoria y la huella de VRAM en un 50%.

Arquitectura:
  Capa 2i   (Generadora) -> Computa Q_(2i), K_(shared), V_(shared).
  Capa 2i+1 (Consumidora) -> Computa únicamente Q_(2i+1) y reutiliza K_(shared), V_(shared).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from .gqa import repeat_kv
from .qk_norm import QKNorm


class CrossLayerAttention(nn.Module):
    """
    Módulo de atención que soporta modo Generador (computa KV) o modo Consumidor
    (reutiliza KV de la capa previa).
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        is_kv_producer: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.is_kv_producer = is_kv_producer

        # Proyección de Query siempre presente
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.qk_norm = QKNorm(self.head_dim)

        # Proyecciones de Key y Value solo si esta capa produce KV
        if self.is_kv_producer:
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        else:
            self.k_proj = None
            self.v_proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        shared_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
            attn_output: [batch, seq_len, hidden_size]
            new_shared_kv: (key_states, value_states) si is_kv_producer=True, o None
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Proyección de Q
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.qk_norm.q_norm(q)

        if self.is_kv_producer:
            k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            k = self.qk_norm.k_norm(k)
            produced_kv = (k, v)
        else:
            if shared_kv is None:
                raise ValueError("La capa consumidora requiere shared_kv de la capa previa.")
            k, v = shared_kv
            produced_kv = None

        # Repetir KV si num_kv_heads < num_heads (GQA)
        k_rep = repeat_kv(k, self.num_heads // self.num_kv_heads)
        v_rep = repeat_kv(v, self.num_heads // self.num_kv_heads)

        # Atención por producto punto escalado
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = torch.matmul(q, k_rep.transpose(-1, -2)) * scale

        # Máscara causal
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device), diagonal=1)
        attn_weights = attn_weights + causal_mask

        probs = torch.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(probs, v_rep)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.o_proj(attn_out)

        return out, produced_kv
