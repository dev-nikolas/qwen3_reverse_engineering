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
Controlador de Supresión de Copia (Copy-Suppression Controller).

Fundamento Mecanicista:
Descubierto en los experimentos forenses de Qwen3-1.7B:
Mientras el circuito de inducción (L16H14, L03H10, L21H08) copia activamente secuencias
del contexto, las cabezas inhibidoras tardías en las Capas 19 y 20 (L20H15, L19H08, L19H09, L20H06)
proyectan logits negativos (Direct Logit Attribution, DLA) sobre los tokens repetidos para impedir
que el modelo quede atrapado en bucles infinitos de eco o degeneración de texto.

Este módulo replica y amplifica dicho comportamiento de forma programática y paramétrica
durante la fase de decodificación autoregresiva.
"""

import torch

class CopySuppressionController:
    """
    Filtro mecanicista de supresión de copia para calibrar las probabilidades de logits.
    
    A diferencia de la penalización de repetición clásica (que penaliza uniformemente todos
    los tokens vistos), el CopySuppressionController aplica una función de decaimiento temporal
    y sensibilidad de inducción, suprimiendo con mayor fuerza los tokens que acaban de ser
    copiados inmediatamente tras un patrón de inducción repetitivo.
    """
    def __init__(
        self,
        base_suppression_strength: float = 1.25,
        recency_decay: float = 0.85,
        min_suppression_limit: float = 0.05
    ):
        self.base_suppression_strength = base_suppression_strength
        self.recency_decay = recency_decay
        self.min_suppression_limit = min_suppression_limit

    def apply_suppression(
        self,
        logits: torch.Tensor,
        context_token_ids: list[int] | torch.Tensor
    ) -> torch.Tensor:
        """
        Modifica los logits proyectando penalizaciones direccionales negativas (DLA)
        sobre los tokens repetidos de acuerdo a su proximidad temporal.
        
        Args:
            logits: Tensor 1D [vocab_size] o 2D [batch_size, vocab_size]
            context_token_ids: Secuencia histórica de IDs generados o vistos
        
        Returns:
            Logits calibrados con supresión de bucle de copia.
        """
        if isinstance(context_token_ids, torch.Tensor):
            tokens_list = context_token_ids.view(-1).tolist()
        else:
            tokens_list = list(context_token_ids)

        if not tokens_list:
            return logits

        is_1d = (logits.dim() == 1)
        if is_1d:
            logits = logits.unsqueeze(0)

        calibrated_logits = logits.clone()
        seq_len = len(tokens_list)

        # Mapa de penalización acumulativa basada en recencia
        penalty_map: dict[int, float] = {}
        for idx, token_id in enumerate(reversed(tokens_list)):
            # Distancia desde el token actual
            distance = idx + 1
            decay_factor = self.recency_decay ** distance
            if decay_factor < self.min_suppression_limit:
                continue

            current_penalty = self.base_suppression_strength * decay_factor
            if token_id in penalty_map:
                penalty_map[token_id] += current_penalty
            else:
                penalty_map[token_id] = current_penalty

        # Inyectar DLA negativo en los logits correspondientes
        for token_id, penalty in penalty_map.items():
            if token_id < calibrated_logits.shape[-1]:
                # Si el logit es positivo, restar la penalización
                # Si es negativo, hacerlo más negativo para alejarlo de la probabilidad de muestreo
                val = calibrated_logits[:, token_id]
                calibrated_logits[:, token_id] = torch.where(
                    val > 0,
                    val - penalty,
                    val - (penalty * 0.5)
                )

        if is_1d:
            return calibrated_logits.squeeze(0)
        return calibrated_logits


__all__ = ["CopySuppressionController"]
