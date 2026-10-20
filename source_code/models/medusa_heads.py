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
Qwen3 Medusa Residual Speculative Heads
========================================
Implementación de cabezales de predicción multi-token residuales (Medusa / MTP)
para decodificación especulativa sin modelo borrador (Zero-Draft Speculative Decoding).

Arquitectura por Cabezal:
  Residual Stream (Capa 28)
           │
     ┌─────┴──────────┐
     │                │
Linear(2048->2048)    │ (Skip Connection)
     │                │
   SiLU               │
     │                │
Linear(2048->2048)    │
     │                │
     └──────► + ◄─────┘
              │
        Linear(2048->Vocab) -> Logits para t + k + 1
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Any


class MedusaResidualBlock(nn.Module):
    """
    Bloque residual no lineal que proyecta el estado oculto del modelo base
    para anticipar representaciones latentes de pasos futuros.
    """
    def __init__(self, hidden_size: int = 2048):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear2(self.act(self.linear1(x)))
        return residual + out


class MedusaHead(nn.Module):
    """
    Cabezal individual de predicción de token para el desplazamiento +k.
    """
    def __init__(self, hidden_size: int = 2048, vocab_size: int = 151936, num_layers: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList([
            MedusaResidualBlock(hidden_size) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)
        return logits


class Qwen3MedusaModel(nn.Module):
    """
    Envoltura para dotar a StandaloneQwen3ForCausalLM de cabezales Medusa
    especulativos nativos.
    
    Args:
        base_model: Modelo causal de Qwen3 (ej. StandaloneQwen3ForCausalLM)
        num_heads: Número de cabezales especulativos (típicamente 3 o 4)
        hidden_size: Dimensión oculta del modelo (2048 en Qwen3-1.7B)
        vocab_size: Tamaño del vocabulario (151936)
    """
    def __init__(
        self,
        base_model: nn.Module,
        num_heads: int = 3,
        hidden_size: int = 2048,
        vocab_size: int = 151936
    ):
        super().__init__()
        self.base_model = base_model
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Cabezales residuales Medusa para t+1, t+2, ..., t+k
        self.medusa_heads = nn.ModuleList([
            MedusaHead(hidden_size=hidden_size, vocab_size=vocab_size)
            for _ in range(num_heads)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass simultáneo:
        Retorna los logits del modelo base (para t+0) y los logits de cada
        cabezal Medusa (para t+1, ..., t+k).
        
        Returns:
            base_logits: [batch, seq_len, vocab_size]
            medusa_logits: Lista de K tensores [batch, seq_len, vocab_size]
        """
        # Obtenemos los logits del modelo base
        base_logits, hidden_states = self._forward_base_with_hidden(input_ids, attention_mask, kv_cache)
        
        # Calcular los logits especulativos de los cabezales Medusa
        medusa_logits = []
        for head in self.medusa_heads:
            medusa_logits.append(head(hidden_states))
            
        return base_logits, medusa_logits

    def _forward_base_with_hidden(self, input_ids, attention_mask, kv_cache):
        """Extrae tanto los logits del cabezal LM como el residual stream final."""
        if hasattr(self.base_model, "model") and hasattr(self.base_model, "lm_head"):
            hidden = self.base_model.model(input_ids, kv_cache=kv_cache)
            base_logits = self.base_model.lm_head(hidden)
            return base_logits, hidden
        else:
            # Fallback en caso de que sea un modelo genérico
            base_logits = self.base_model(input_ids)
            return base_logits, base_logits  # Aproximación si no expone estados ocultos

    def generate_candidates(
        self,
        base_logits: torch.Tensor,
        medusa_logits: List[torch.Tensor],
        top_k: int = 1
    ) -> List[torch.Tensor]:
        """
        Genera la secuencia candidata especulativa para la última posición generada.
        
        Returns:
            candidates: Lista de K+1 tokens predichos [base_token, medusa_1, medusa_2, ...]
        """
        # Predicción del modelo base para la posición actual
        last_base_logit = base_logits[:, -1, :]  # [batch, vocab]
        base_cand = torch.argmax(last_base_logit, dim=-1, keepdim=True)
        candidates = [base_cand]

        # Predicciones de los cabezales Medusa para pasos sucesivos
        for head_logit in medusa_logits:
            last_head_logit = head_logit[:, -1, :]
            cand = torch.argmax(last_head_logit, dim=-1, keepdim=True)
            candidates.append(cand)

        return candidates
