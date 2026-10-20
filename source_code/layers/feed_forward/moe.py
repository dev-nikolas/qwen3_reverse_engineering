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
Módulo de Mezcla de Expertos Dispersa (Sparse Mixture of Experts / MoE) para Qwen3.

Fundamento de Ingeniería Inversa:
En las variantes MoE de la familia Qwen (como Qwen3-MoE y DeepSeek/Qwen híbridos):
1. Router Top-K Disperso: Enruta cada token hacia los 'top_k' expertos más afines (típicamente 2 de N).
2. Expertos Compartidos (Shared Experts): Expertos fijos que procesan TODOS los tokens sin
   pasar por el router, anclando el conocimiento general y evitando la pérdida de coherencia común
   en arquitecturas MoE estándar.
3. Expertos SwiGLU: Cada experto individual implementa la proyección no lineal SwiGLU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swiglu import SwiGLU

class Qwen3MoeRouter(nn.Module):
    """
    Enrutador Top-K para asignar tokens a expertos específicos.
    """
    def __init__(self, hidden_size: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch * seq_len, hidden_size]
        Returns:
            routing_weights: [batch * seq_len, top_k] normalizado por softmax
            selected_experts: [batch * seq_len, top_k] índices de expertos
            router_logits: [batch * seq_len, num_experts] para loss de balanceo
        """
        router_logits = self.gate(x) # [total_tokens, num_experts]
        
        # Seleccionar top-k expertos
        top_k_logits, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        # Normalizar con softmax sobre los top-k seleccionados
        routing_weights = F.softmax(top_k_logits, dim=-1, dtype=torch.float32).to(x.dtype)
        
        return routing_weights, selected_experts, router_logits


class Qwen3SparseMoeBlock(nn.Module):
    """
    Bloque SwiGLU MoE completo con Expertos Enrutados y Expertos Compartidos.
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 1024,
        num_experts: int = 8,
        num_shared_experts: int = 1,
        top_k: int = 2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts

        # Router
        self.router = Qwen3MoeRouter(hidden_size, num_experts=num_experts, top_k=top_k)

        # Expertos Enrutados
        self.experts = nn.ModuleList([
            SwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size)
            for _ in range(num_experts)
        ])

        # Expertos Compartidos (Shared Experts)
        if num_shared_experts > 0:
            shared_intermediate = intermediate_size * num_shared_experts
            self.shared_expert = SwiGLU(hidden_size=hidden_size, intermediate_size=shared_intermediate)
        else:
            self.shared_expert = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor [batch, seq_len, hidden_size]
        Returns:
            output: Tensor [batch, seq_len, hidden_size]
            aux_loss: Pérdida auxiliar de balanceo de carga
        """
        batch, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        routing_weights, selected_experts, router_logits = self.router(x_flat)
        
        # Contenedor para acumular la salida ponderada
        final_output = torch.zeros_like(x_flat)

        # Cómputo disperso por experto
        for expert_idx, expert_layer in enumerate(self.experts):
            # Encontrar tokens asignados a este experto
            # mask: [total_tokens, top_k]
            mask = (selected_experts == expert_idx)
            token_indices, k_positions = torch.where(mask)

            if token_indices.numel() > 0:
                tokens_for_expert = x_flat[token_indices]
                expert_out = expert_layer(tokens_for_expert)
                weights_for_expert = routing_weights[token_indices, k_positions].unsqueeze(-1)
                
                # Acumular ponderado
                final_output.index_add_(0, token_indices, expert_out * weights_for_expert)

        # Contribución aditiva de los expertos compartidos
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)
            final_output = final_output + shared_out

        # Pérdida auxiliar de balanceo de carga (Auxiliary Load Balancing Loss)
        # L = num_experts * sum(f_i * P_i)
        probs = F.softmax(router_logits, dim=-1)
        freqs = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            freqs[i] = (selected_experts == i).float().mean()
        mean_probs = probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(freqs * mean_probs)

        return final_output.view(batch, seq_len, hidden_size), aux_loss


__all__ = ["Qwen3SparseMoeBlock", "Qwen3MoeRouter"]
