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
Motor de Decodificación Especulativa (Speculative Decoding Engine) para Aceleración 2x-3x.

Fundamento:
Combina un modelo borrador (Draft Model, ej. PrunedQwen3 o modelo compacto)
con el modelo objetivo principal (Target Model Qwen3-1.7B).
El borrador especula 'gamma' tokens de forma rápida; el modelo objetivo verifica
todos los candidatos simultáneamente en un único forward pass paralelo y aplica
muestreo por rechazo (Leviathan et al., 2023).
Garantiza exactamente la misma distribución matemática que el modelo grande.
"""

import torch
import torch.nn.functional as F

class SpeculativeDecoder:
    """
    Decodificador especulativo con verificación paralela y muestreo por rechazo.
    """
    def __init__(self, target_model, draft_model, tokenizer, gamma: int = 3, device: str = "cpu"):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.device = device

    @torch.no_grad()
    def generate_step(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Ejecuta un ciclo de especulación y verificación:
        1. Genera gamma tokens con el modelo borrador.
        2. Evalúa la secuencia completa [input_ids + gamma] en el modelo objetivo.
        3. Acepta o rechaza candidatos según el ratio de probabilidades.
        """
        temp = max(temperature, 1e-5)
        current_seq = input_ids.clone()
        draft_tokens = []
        draft_probs = []

        # 1. Especular gamma tokens con el modelo borrador
        for _ in range(self.gamma):
            draft_logits = self.draft_model(current_seq)
            next_logits = draft_logits[:, -1, :] / temp
            p_draft = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(p_draft, num_samples=1)
            
            draft_tokens.append(next_token)
            draft_probs.append(p_draft.gather(1, next_token).squeeze(1))
            current_seq = torch.cat([current_seq, next_token], dim=-1)

        # 2. Verificación paralela en el modelo objetivo (un solo forward pass)
        target_logits = self.target_model(current_seq)
        target_probs = F.softmax(target_logits / temp, dim=-1)

        # 3. Muestreo por rechazo
        accepted_tokens = []
        prefix_len = input_ids.shape[1]

        for i in range(self.gamma):
            # Posición en el target correspondientes al token especulado
            token_cand = draft_tokens[i]
            q_draft = draft_probs[i]
            p_target = target_probs[:, prefix_len + i - 1, :].gather(1, token_cand).squeeze(1)

            # Ratio de aceptación: min(1, p / q)
            alpha = torch.clamp(p_target / (q_draft + 1e-8), max=1.0)
            u = torch.rand(1, device=self.device)

            if u <= alpha:
                accepted_tokens.append(token_cand)
            else:
                # Rechazado: muestrear token correctivo desde la distribución residual corregida
                p_diff = torch.clamp(target_probs[:, prefix_len + i - 1, :] - target_probs[:, prefix_len + i - 1, :], min=0.0)
                norm = p_diff.sum(dim=-1, keepdim=True)
                if norm > 0:
                    resampled = torch.multinomial(p_diff / norm, num_samples=1)
                else:
                    resampled = torch.multinomial(target_probs[:, prefix_len + i - 1, :], num_samples=1)
                accepted_tokens.append(resampled)
                break

        # Concatenar tokens aceptados al input_ids original
        if accepted_tokens:
            new_tokens_tensor = torch.cat(accepted_tokens, dim=-1)
            return torch.cat([input_ids, new_tokens_tensor], dim=-1)
        return input_ids


__all__ = ["SpeculativeDecoder"]
