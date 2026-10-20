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
Qwen3 Reasoning & Alignment Loss Engine
========================================
Motor de pérdidas matemáticas y funciones de recompensa para el entrenamiento
de razonamiento de cadena de pensamiento (<think> ... </think>) en Qwen3.

Implementa:
1. GRPOLoss: Group Relative Policy Optimization (sin red de valor/crítico externa),
   optimizando ventajas estandarizadas dentro de un grupo de trayectorias CoT.
2. DPOLoss: Direct Preference Optimization con divergencia KL implícita para
   calibrar trayectorias aceptadas (chosen) vs rechazadas (rejected).
3. ReasoningFormatReward: Función de verificación estricta de sintaxis XML y
   penalización por desborde de presupuesto de razonamiento (budget overrun).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class ReasoningFormatReward:
    """
    Evaluador de formato para la cadena de pensamiento CoT (<think>...</think>).
    Otorga recompensa +1.0 por estructura válida, penaliza la ausencia de cierre
    y aplica penalización suave si la longitud excede el presupuesto límite.
    """
    def __init__(
        self,
        think_start: str = "<think>",
        think_end: str = "</think>",
        max_budget_tokens: int = 1024,
        budget_penalty_coef: float = 0.001
    ):
        self.think_start = think_start
        self.think_end = think_end
        self.max_budget_tokens = max_budget_tokens
        self.budget_penalty_coef = budget_penalty_coef

    def score_text(self, text: str, token_length: int) -> float:
        reward = 0.0
        
        # 1. Presencia de etiquetas
        has_start = self.think_start in text
        has_end = self.think_end in text
        
        if has_start and has_end:
            start_pos = text.find(self.think_start)
            end_pos = text.find(self.think_end)
            if start_pos < end_pos:
                reward += 1.0  # Estructura perfectamente anidada
            else:
                reward -= 1.0  # Orden invertido
        elif has_start and not has_end:
            reward -= 0.8  # Razonamiento truncado sin conclusión
        elif not has_start and has_end:
            reward -= 0.5  # Cierre huérfano

        # 2. Penalización por exceso de presupuesto de tokens
        if token_length > self.max_budget_tokens:
            excess = token_length - self.max_budget_tokens
            reward -= self.budget_penalty_coef * excess

        return reward


class GRPOLoss(nn.Module):
    """
    Group Relative Policy Optimization (GRPO) Loss.
    
    Elimina la necesidad de un modelo crítico (Value Model de 1.7B params),
    reduciendo el consumo de memoria VRAM en 50% durante el RLHF.
    Normaliza las recompensas de $G$ muestras generadas para la misma entrada:
        A_i = (r_i - mean(r)) / (std(r) + eps)
    """
    def __init__(self, clip_eps: float = 0.2, beta_kl: float = 0.04):
        super().__init__()
        self.clip_eps = clip_eps
        self.beta_kl = beta_kl

    def forward(
        self,
        log_probs: torch.Tensor,        # [G, seq_len] log pi_theta(y|x)
        old_log_probs: torch.Tensor,    # [G, seq_len] log pi_old(y|x)
        ref_log_probs: torch.Tensor,    # [G, seq_len] log pi_ref(y|x)
        rewards: torch.Tensor,          # [G] recompensas escalares por muestra
        mask: Optional[torch.Tensor] = None # [G, seq_len] máscara de tokens de respuesta
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcula la pérdida PPO recortada basada en grupos y la divergencia KL.
        """
        G = rewards.shape[0]
        # 1. Calcular ventajas relativas estandarizadas dentro del grupo
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r  # [G]
        advantages = advantages.unsqueeze(1)    # [G, 1]

        # 2. Ratio de probabilidad de política: pi_theta / pi_old
        # Sumar o promediar log-probabilidades a nivel de secuencia
        if mask is not None:
            seq_log_prob = (log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
            seq_old_log_prob = (old_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
            seq_ref_log_prob = (ref_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        else:
            seq_log_prob = log_probs.mean(dim=-1)
            seq_old_log_prob = old_log_probs.mean(dim=-1)
            seq_ref_log_prob = ref_log_probs.mean(dim=-1)

        ratio = torch.exp(seq_log_prob - seq_old_log_prob).unsqueeze(1)  # [G, 1]

        # 3. Función objetivo PPO recortada
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 4. Divergencia KL analítica contra el modelo de referencia (evita colapso)
        # D_KL ~ exp(ref - theta) - (ref - theta) - 1 o aproximación (pi_theta - pi_ref)
        kl = (seq_log_prob - seq_ref_log_prob)
        kl_loss = self.beta_kl * kl.mean()

        total_loss = policy_loss + kl_loss

        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_reward": mean_r.item(),
            "std_reward": std_r.item()
        }
        return total_loss, metrics


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss.
    Optimiza el contraste entre cadenas de razonamiento elegidas (chosen) y
    rechazadas (rejected) directamente sobre los logits de la política.
    """
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        pi_chosen_logps: torch.Tensor,    # [batch] log pi(y_chosen | x)
        pi_rejected_logps: torch.Tensor,  # [batch] log pi(y_rejected | x)
        ref_chosen_logps: torch.Tensor,   # [batch] log pi_ref(y_chosen | x)
        ref_rejected_logps: torch.Tensor  # [batch] log pi_ref(y_rejected | x)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retorna:
            loss: Escalar de pérdida DPO
            chosen_rewards: Recompensa implícita del par ganador
            rejected_rewards: Recompensa implícita del par perdedor
        """
        pi_logratios = pi_chosen_logps - pi_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios
        scaled_logits = self.beta * logits

        # Pérdida sigmoidal con label smoothing opcional
        losses = (
            -F.logsigmoid(scaled_logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-scaled_logits) * self.label_smoothing
        )
        loss = losses.mean()

        chosen_rewards = self.beta * (pi_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (pi_rejected_logps - ref_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards
