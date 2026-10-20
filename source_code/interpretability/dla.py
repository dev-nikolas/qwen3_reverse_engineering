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
Descomposición y Análisis de Atribución Directa de Logits (Direct Logit Attribution / DLA).

Fundamento Mecanicista:
El flujo residual es lineal y acumulativo:
    h_final = embed + sum_{l=0}^{27} (Attn_l + MLP_l)
Al proyectar h_final al vocabulario con W_U, la contribución neta a un token 'y' se puede
descomponer aditivamente en las contribuciones individuales de cada cabeza y cada bloque MLP:
    Logit(y) = W_U[y] . embed + sum_{l} W_U[y] . Attn_l + sum_{l} W_U[y] . MLP_l

Este módulo cuantifica qué componentes específicos de Qwen3 son los causantes directos
de promover o inhibir cualquier token candidato.
"""

import torch
import torch.nn as nn

class DirectLogitAttributionAnalyzer:
    """
    Descompone las contribuciones directas de capas y cabezas a los logits finales.
    """
    def __init__(self, unembedding_matrix: torch.Tensor):
        """
        Args:
            unembedding_matrix: Tensor de desembebing W_U de forma [vocab_size, hidden_size]
        """
        self.W_U = unembedding_matrix.detach()

    def compute_component_dla(
        self,
        component_output: torch.Tensor,
        target_token_id: int,
        baseline_token_id: int | None = None
    ) -> float:
        """
        Calcula la contribución de un componente al logit del token objetivo (o al logit residual target - baseline).
        
        Args:
            component_output: Tensor [hidden_size] correspondiente al último token emitido por el componente.
            target_token_id: ID del token a promover
            baseline_token_id: ID del token competidor (opcional)
        """
        vec = component_output.view(-1).float()
        w_target = self.W_U[target_token_id].float()

        if baseline_token_id is not None:
            w_baseline = self.W_U[baseline_token_id].float()
            diff_direction = w_target - w_baseline
            return float(torch.dot(vec, diff_direction).item())
        else:
            return float(torch.dot(vec, w_target).item())

    def rank_heads_by_attribution(
        self,
        head_outputs: list[torch.Tensor],
        target_token_id: int,
        baseline_token_id: int | None = None
    ) -> list[dict]:
        """
        Ordena un catálogo de cabezas de acuerdo a su atribución de logits directa.
        
        Args:
            head_outputs: Lista de tensores [hidden_size]
        """
        scores = []
        for idx, h_out in enumerate(head_outputs):
            attr = self.compute_component_dla(h_out, target_token_id, baseline_token_id)
            scores.append({
                "component_idx": idx,
                "attribution": round(attr, 4)
            })
        scores.sort(key=lambda x: x["attribution"], reverse=True)
        return scores


__all__ = ["DirectLogitAttributionAnalyzer"]
