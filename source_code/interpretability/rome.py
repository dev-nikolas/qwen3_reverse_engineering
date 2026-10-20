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
Qwen3 ROME (Rank-One Model Editing) & Knowledge Surgery Engine
==============================================================
Motor forense de edición quirúrgica de memorias y hechos factuales en las capas
intermedias del MLP de Qwen3 (Capas 14 a 18) sin reentrenamiento.

Fundamento Matemático (ROME):
  W_new = W + Delta
  Delta = (v_star - W * k_star) * (C_inv * k_star)^T / ((k_star)^T * C_inv * k_star)

Donde:
  - k_star: Vector clave de activación del sujeto en la capa seleccionada.
  - v_star: Vector valor objetivo que induce el nuevo concepto deseado.
  - C_inv: Inversa de la matriz de covarianza de activaciones del residual stream.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class ROMEKnowledgeEditor:
    """
    Editor de conocimiento de rango 1 para matrices de proyección del MLP de Qwen3.
    """
    def __init__(self, hidden_size: int = 2048, cov_epsilon: float = 1e-4):
        self.hidden_size = hidden_size
        self.cov_epsilon = cov_epsilon

    def compute_rank_one_update(
        self,
        weight: torch.Tensor,       # [out_features, in_features]
        k_star: torch.Tensor,       # [in_features] Vector clave del sujeto
        v_star: torch.Tensor,       # [out_features] Vector valor objetivo
        C_inv: Optional[torch.Tensor] = None # [in_features, in_features] Matriz de covarianza inversa
    ) -> torch.Tensor:
        """
        Calcula la perturbación de matriz Delta de rango 1 exacta.
        """
        device = weight.device
        dtype = weight.dtype

        k = k_star.to(device=device, dtype=torch.float32)
        v = v_star.to(device=device, dtype=torch.float32)
        W = weight.to(device=device, dtype=torch.float32)

        # Si no se provee covarianza, se asume covarianza isotrópica (C = I)
        if C_inv is None:
            c_inv_k = k
            denom = torch.dot(k, k).clamp(min=1e-6)
        else:
            C_inv_f = C_inv.to(device=device, dtype=torch.float32)
            c_inv_k = torch.matmul(C_inv_f, k)
            denom = torch.dot(k, c_inv_k).clamp(min=1e-6)

        # Error residual: (v_star - W * k_star)
        diff = v - torch.matmul(W, k)

        # Delta = diff * (c_inv_k)^T / denom
        delta = torch.outer(diff, c_inv_k) / denom

        return delta.to(dtype=dtype)

    def apply_edit_to_layer(
        self,
        linear_module: nn.Linear,
        k_star: torch.Tensor,
        v_star: torch.Tensor,
        C_inv: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Aplica directamente la actualización ROME sobre una capa lineal del modelo.
        """
        with torch.no_grad():
            delta = self.compute_rank_one_update(
                linear_module.weight.data,
                k_star,
                v_star,
                C_inv
            )
            # Aplicar modificación in-place
            linear_module.weight.data.add_(delta)

            norm_delta = torch.norm(delta).item()
            norm_orig = torch.norm(linear_module.weight.data).item()

        return {
            "delta_norm": norm_delta,
            "weight_norm": norm_orig,
            "relative_change": norm_delta / max(norm_orig, 1e-6)
        }
