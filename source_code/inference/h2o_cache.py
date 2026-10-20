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
H2O (Heavy-Hitter Oracle) Dynamic KV-Cache Compactor
=====================================================
Compactor de KV-Cache basado en la política Heavy-Hitter Oracle (H2O) para Qwen3.

Observación Clave:
  La masa de atención en contextos largos está concentrada en una ley potencial
  (power-law): una fracción mínima de tokens acumula la mayoría del peso atencional
  (tokens clave / heavy-hitters), sumados a los sumideros iniciales (Attention Sinks)
  y la ventana local reciente.

Estructura del Buffer Dinámico:
  [Sinks (0..S)] + [Heavy Hitters (Top H)] + [Ventana Local Reciente (L)]
"""

import torch
from typing import Tuple, Optional, Set


class H2OKVCache:
    """
    Buffer de claves y valores con política de desalojo inteligente H2O.
    """
    def __init__(
        self,
        num_sinks: int = 4,
        max_heavy_hitters: int = 64,
        local_window_size: int = 64
    ):
        self.num_sinks = num_sinks
        self.max_heavy_hitters = max_heavy_hitters
        self.local_window_size = local_window_size
        self.max_capacity = num_sinks + max_heavy_hitters + local_window_size

        self.k: Optional[torch.Tensor] = None  # [batch, kv_heads, seq_len, head_dim]
        self.v: Optional[torch.Tensor] = None
        self.cumulative_scores: Optional[torch.Tensor] = None  # [batch, seq_len]

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        attn_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Incorpora nuevos tokens al KV-cache y desaloja tokens no críticos
        si se supera la capacidad máxima.
        
        Args:
            new_k: [batch, kv_heads, new_len, head_dim]
            new_v: [batch, kv_heads, new_len, head_dim]
            attn_scores: [batch, heads, q_len, kv_len] (opcional)
        """
        batch_size = new_k.shape[0]

        if self.k is None:
            self.k = new_k
            self.v = new_v
            self.cumulative_scores = torch.zeros((batch_size, new_k.shape[-2]), device=new_k.device)
        else:
            self.k = torch.cat([self.k, new_k], dim=-2)
            self.v = torch.cat([self.v, new_v], dim=-2)
            new_scores = torch.zeros((batch_size, new_k.shape[-2]), device=new_k.device)
            self.cumulative_scores = torch.cat([self.cumulative_scores, new_scores], dim=-1)

        # Actualizar acumulador de atenciones si se suministraron los pesos
        if attn_scores is not None:
            # Promediar sobre cabezas y longitud de consulta: [batch, kv_len]
            step_scores = attn_scores.mean(dim=(1, 2))
            min_len = min(self.cumulative_scores.shape[-1], step_scores.shape[-1])
            self.cumulative_scores[:, :min_len] += step_scores[:, :min_len]

        # Comprobar desalojo si la longitud total excede la capacidad
        total_tokens = self.k.shape[-2]
        if total_tokens > self.max_capacity:
            self._evict_low_utility_tokens()

        return self.k, self.v

    def _evict_low_utility_tokens(self):
        """
        Aplica la política H2O: preserva sinks, selecciona los top heavy-hitters
        en la región intermedia, y retiene la ventana local reciente.
        """
        total_tokens = self.k.shape[-2]
        sink_indices = list(range(min(self.num_sinks, total_tokens)))
        local_start = max(0, total_tokens - self.local_window_size)
        local_indices = list(range(local_start, total_tokens))

        # Candidatos a Heavy-Hitters en la región media
        middle_candidates = list(set(range(total_tokens)) - set(sink_indices) - set(local_indices))
        
        if middle_candidates and self.max_heavy_hitters > 0:
            cand_tensor = torch.tensor(middle_candidates, device=self.cumulative_scores.device)
            # Tomar los scores acumulados medios a través del lote
            cand_scores = self.cumulative_scores.mean(dim=0)[cand_tensor]
            
            top_k = min(self.max_heavy_hitters, len(middle_candidates))
            top_indices_in_cands = torch.topk(cand_scores, k=top_k).indices
            heavy_hitter_indices = cand_tensor[top_indices_in_cands].tolist()
        else:
            heavy_hitter_indices = []

        # Unión ordenada de índices retenidos
        kept_indices = sorted(list(set(sink_indices + heavy_hitter_indices + local_indices)))
        idx_tensor = torch.tensor(kept_indices, device=self.k.device, dtype=torch.long)

        # Compactar tensores
        self.k = torch.index_select(self.k, dim=-2, index=idx_tensor)
        self.v = torch.index_select(self.v, dim=-2, index=idx_tensor)
        self.cumulative_scores = torch.index_select(self.cumulative_scores, dim=-1, index=idx_tensor)
