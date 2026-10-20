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
Qwen3 MoE Expert Specialization & Router Saliency Analyzer
==========================================================
Analizador mecanicista de especialización de expertos y compuertas de enrutamiento
para arquitecturas Sparse Mixture-of-Experts (MoE) en Qwen3.

Capacidades:
1. Router Gate Saliency: Calcula la distribución de masa de probabilidad
   P(expert = e | x) a lo largo de representaciones de texto token por token.
2. Expert Specialization Index: Mide la selectividad de cada experto ante dominios
   específicos (matemáticas, código, razonamiento CoT, sintaxis general).
3. Dead Expert Detection: Identifica expertos no activados o colapso de enrutador
   (routing collapse) que degradan el balance de carga.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any


class ExpertSaliencyAnalyzer:
    """
    Analizador de enrutamiento y saliencia de expertos para Qwen3 MoE.
    """
    def __init__(self, num_experts: int, top_k: int = 2):
        self.num_experts = num_experts
        self.top_k = top_k
        self.domain_activations: Dict[str, torch.Tensor] = {}

    def compute_routing_stats(
        self,
        router_logits: torch.Tensor,
        top_k_indices: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Calcula estadísticas de uso y entropía de las compuertas de enrutamiento.
        
        Args:
            router_logits: [batch_size, seq_len, num_experts]
            top_k_indices: [batch_size, seq_len, top_k]
        """
        batch_size, seq_len, _ = router_logits.shape
        total_tokens = batch_size * seq_len
        
        # Probabilidades completas del softmax
        probs = torch.softmax(router_logits.float(), dim=-1).view(-1, self.num_experts)
        
        # 1. Carga fraccional de cada experto (cuántos tokens lo tienen en su Top-K)
        flat_indices = top_k_indices.view(-1)
        expert_counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
        expert_load = expert_counts / (total_tokens * self.top_k)

        # 2. Entropía del enrutador (alta entropía = balance uniforme; baja = alta especialización)
        mean_probs = probs.mean(dim=0)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10)).item()

        # 3. Detección de expertos inactivos (carga < 1%)
        dead_experts = [idx for idx, load in enumerate(expert_load.tolist()) if load < 0.01]

        return {
            "expert_load": expert_load.tolist(),
            "mean_expert_probs": mean_probs.tolist(),
            "router_entropy": entropy,
            "dead_experts": dead_experts,
            "total_tokens_evaluated": total_tokens
        }

    def record_domain_routing(
        self,
        domain_name: str,
        router_probs: torch.Tensor
    ):
        """
        Registra la distribución de compuertas ante un dominio conceptual dado.
        
        Args:
            domain_name: Nombre del dominio (ej. 'math', 'code', 'reasoning', 'chat')
            router_probs: [tokens, num_experts] probabilidades softmax
        """
        probs_2d = router_probs.view(-1, self.num_experts).detach().cpu()
        mean_profile = probs_2d.mean(dim=0)
        self.domain_activations[domain_name] = mean_profile

    def compute_domain_specialization_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Calcula una matriz de correlación entre dominios y expertos activados.
        """
        matrix = {}
        for domain, profile in self.domain_activations.items():
            matrix[domain] = {
                f"expert_{i}": round(profile[i].item(), 4)
                for i in range(self.num_experts)
            }
        return matrix
