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
Latent Uncertainty & Real-Time Hallucination Detector
=====================================================
Monitor mecanicista de detección temprana de alucinaciones y divergencia latente
en Qwen3 basado en dinámica de entropía y deriva en el flujo residual.

Métricas Evaluadas:
1. Logit Lens Layerwise Entropy: Entropía de Shannon H(p_l) al desproyectar
   estados intermedios sobre el vocabulario mediante W_U.
2. Residual Drift Angle: Cambio angular cos(theta) entre capas consecutivas;
   un giro abrupto en capas tardías (22-26) suele anticipar alucinación.
3. Hallucination Risk Score: Índice ponderado normalizado en [0.0, 1.0].
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any


class HallucinationDetector:
    """
    Detector de riesgo de alucinación y monitoreo de certeza latente.
    """
    def __init__(self, unembedding_weight: torch.Tensor, entropy_threshold: float = 3.5):
        """
        Args:
            unembedding_weight: Matriz W_U [vocab_size, hidden_size]
            entropy_threshold: Umbral de entropía para alertar sobre incertidumbre
        """
        self.W_U = unembedding_weight.float()
        self.entropy_threshold = entropy_threshold

    def compute_entropy_from_hidden(self, hidden_state: torch.Tensor) -> float:
        """
        Calcula la entropía de Shannon al proyectar un estado oculto a través de W_U.
        
        Args:
            hidden_state: Vector [hidden_size]
        """
        logits = torch.matmul(self.W_U, hidden_state.float())
        probs = torch.softmax(logits, dim=-1)
        # H = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12))).item()
        return entropy

    def compute_residual_drift(
        self,
        hidden_l: torch.Tensor,
        hidden_l_next: torch.Tensor
    ) -> float:
        """
        Calcula la distancia angular (1 - cosine_similarity) entre capas contiguas.
        """
        sim = torch.nn.functional.cosine_similarity(
            hidden_l.unsqueeze(0).float(),
            hidden_l_next.unsqueeze(0).float()
        ).item()
        return max(0.0, 1.0 - sim)

    def analyze_token_trajectory(
        self,
        layer_hidden_states: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Analiza la trayectoria de un token a lo largo de las 28 capas del modelo.
        
        Args:
            layer_hidden_states: Lista de tensores [hidden_size] de cada capa (0 a 27)
        """
        n_layers = len(layer_hidden_states)
        entropies = [self.compute_entropy_from_hidden(h) for h in layer_hidden_states]
        
        drifts = []
        for i in range(n_layers - 1):
            drifts.append(self.compute_residual_drift(layer_hidden_states[i], layer_hidden_states[i + 1]))

        # Entropía en capas medias-tardías (donde se consolidan hechos semánticos)
        mid_late_start = int(n_layers * 0.5)
        late_entropies = entropies[mid_late_start:]
        mean_late_entropy = sum(late_entropies) / len(late_entropies)

        # Deriva en las últimas 4 capas
        late_drifts = drifts[-4:] if len(drifts) >= 4 else drifts
        mean_late_drift = sum(late_drifts) / max(len(late_drifts), 1)

        # Cálculo de Score de Riesgo (0.0 a 1.0)
        # Factor de entropía relativo
        entropy_factor = min(1.0, mean_late_entropy / (self.entropy_threshold * 1.5))
        drift_factor = min(1.0, mean_late_drift * 5.0)

        risk_score = 0.6 * entropy_factor + 0.4 * drift_factor
        risk_score = max(0.0, min(1.0, risk_score))

        return {
            "hallucination_risk_score": round(risk_score, 4),
            "is_hallucinating": risk_score > 0.65,
            "mean_late_entropy": round(mean_late_entropy, 4),
            "mean_late_drift": round(mean_late_drift, 4),
            "trajectory_entropies": [round(e, 2) for e in entropies]
        }
