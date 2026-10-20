# Copyright 2025 The Qwen Team and Alibaba Group.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Module: Monte Carlo Tree Search (MCTS) for Test-Time Compute Reasoning
# Implements MCTS search trees, UCT branch selection, step-level PRM scoring,
# trajectory rollouts and pruning for Qwen3 mathematical and CoT reasoning.

import math
import random
from typing import List, Dict, Any, Optional, Callable
import torch
import torch.nn as nn

try:
    from source_code.training.reasoning_verifier import StepLevelPRM, ReasoningVerifier
except ImportError:
    from training.reasoning_verifier import StepLevelPRM, ReasoningVerifier


class MCTSNode:
    """
    Nodo individual en el árbol de razonamiento Monte Carlo.
    Representa un paso intermedio de deducción o derivación formal.
    """
    def __init__(
        self,
        state_text: str,
        step_index: int = 0,
        parent: Optional["MCTSNode"] = None,
        prior_prob: float = 1.0,
        step_reward: float = 0.0,
        is_terminal: bool = False
    ):
        self.state_text = state_text
        self.step_index = step_index
        self.parent = parent
        self.prior_prob = float(prior_prob)
        self.step_reward = float(step_reward)
        self.is_terminal = is_terminal
        
        self.children: List["MCTSNode"] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0

    @property
    def value(self) -> float:
        """Valor promedio de retorno acumulado Q(s, a)."""
        if self.visit_count == 0:
            return self.step_reward
        return self.total_value / self.visit_count

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def uct_score(self, c_puct: float = 1.414) -> float:
        """
        Cálculo de UCT (Upper Confidence bounds applied to Trees) con sesgo de probabilidad previa:
        UCT = Q(s, a) + c_puct * P(s, a) * sqrt(N(parent)) / (1 + N(s, a))
        """
        if self.parent is None:
            return self.value

        parent_visits = max(1, self.parent.visit_count)
        exploration = c_puct * self.prior_prob * (math.sqrt(parent_visits) / (1 + self.visit_count))
        return self.value + exploration

    def best_child(self, c_puct: float = 1.414) -> "MCTSNode":
        """Selecciona el hijo con el puntaje UCT más alto."""
        if not self.children:
            raise ValueError("No child nodes to select from.")
        return max(self.children, key=lambda node: node.uct_score(c_puct))

    def get_full_trajectory(self) -> List[str]:
        """Reconstruye la secuencia de pasos desde la raíz hasta este nodo."""
        path = []
        curr = self
        while curr is not None:
            if curr.state_text:
                path.append(curr.state_text)
            curr = curr.parent
        path.reverse()
        return path


class Qwen3MCTSSearchEngine:
    """
    Motor MCTS para escalado de cómputo en inferencia (Test-Time Compute Scaling).
    Orquesta la exploración sistemática de razonamiento:
    1. Selección: desciende por el árbol guiado por UCT.
    2. Expansión: genera K pasos de razonamiento candidatos mediante la política de muestreo.
    3. Evaluación: evalúa la verosimilitud de cada paso mediante el Process Reward Model (StepLevelPRM).
    4. Retropropagación: actualiza visitas y valores hacia la raíz.
    5. Poda: poda ramas con recompensa inferior al umbral de confianza mínimo.
    """
    def __init__(
        self,
        prm_model: Optional[StepLevelPRM] = None,
        c_puct: float = 1.414,
        prune_threshold: float = 0.25,
        max_depth: int = 8,
        branch_factor: int = 3
    ):
        self.prm_model = prm_model
        self.c_puct = c_puct
        self.prune_threshold = prune_threshold
        self.max_depth = max_depth
        self.branch_factor = branch_factor
        self.verifier = ReasoningVerifier()

    def search(
        self,
        root_prompt: str,
        num_simulations: int = 20,
        step_generator: Optional[Callable[[str, int], List[Tuple[str, float]]]] = None,
        step_embedder: Optional[Callable[[str], torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta la búsqueda MCTS completa sobre el problema planteado.
        
        Args:
            root_prompt: Pregunta o premisa inicial.
            num_simulations: Número total de simulaciones Monte Carlo a realizar.
            step_generator: Función generadora (current_trajectory_text, k) -> List[(step_text, prior_prob)].
            step_embedder: Función que extrae tensor (1, hidden_size) para evaluación con PRM.
        """
        root = MCTSNode(state_text=root_prompt, step_index=0)

        # Fallback de generador por defecto para simulaciones sintéticas/pruebas
        if step_generator is None:
            def default_generator(context: str, k: int):
                return [
                    (f"Deducción lógica paso #{random.randint(100, 999)} con derivación.", 1.0 / k)
                    for _ in range(k)
                ]
            step_generator = default_generator

        for sim_idx in range(num_simulations):
            # 1. Selección
            node = root
            while not node.is_leaf() and not node.is_terminal:
                node = node.best_child(self.c_puct)

            # Si alcanzó profundidad máxima o es terminal, saltar expansión
            if node.step_index >= self.max_depth or node.is_terminal:
                reward = node.step_reward
            else:
                # 2. Expansión
                candidates = step_generator("\n".join(node.get_full_trajectory()), self.branch_factor)
                
                # Evaluar pasos candidatos con PRM si está disponible
                for step_txt, prior in candidates:
                    step_reward = 0.5  # Recompensa base neutral
                    is_term = "</think>" in step_txt or "Q.E.D." in step_txt or "Respuesta:" in step_txt

                    if self.prm_model is not None and step_embedder is not None:
                        with torch.no_grad():
                            emb = step_embedder(step_txt)
                            if emb.dim() == 2:
                                emb = emb.unsqueeze(0)  # (1, 1, H)
                            step_reward = float(self.prm_model(emb).squeeze().item())
                    elif self.prm_model is not None:
                        # Fallback PRM sintético si no hay embedder
                        step_reward = max(0.1, min(0.99, prior + random.uniform(-0.1, 0.1)))

                    # 3. Poda estricta de ramas espurias
                    if step_reward >= self.prune_threshold or is_term:
                        child = MCTSNode(
                            state_text=step_txt,
                            step_index=node.step_index + 1,
                            parent=node,
                            prior_prob=prior,
                            step_reward=step_reward,
                            is_terminal=is_term
                        )
                        node.children.append(child)

                # Si ningún hijo superó la poda, evaluar nodo actual
                reward = max([c.step_reward for c in node.children], default=node.step_reward)

            # 4. Retropropagación (Backpropagation)
            curr = node
            while curr is not None:
                curr.visit_count += 1
                curr.total_value += reward
                curr = curr.parent

        # 5. Extracción de la trayectoria ganadora (nodo con mayor visit_count o valor medio)
        best_trajectory = self._extract_best_trajectory(root)
        
        return {
            "root_prompt": root_prompt,
            "total_simulations": num_simulations,
            "best_trajectory": best_trajectory,
            "best_score": best_trajectory[-1].value if best_trajectory else 0.0,
            "formatted_solution": self._format_solution(root_prompt, best_trajectory)
        }

    def _extract_best_trajectory(self, root: MCTSNode) -> List[MCTSNode]:
        """Sigue el camino más visitado desde la raíz hasta la hoja terminal."""
        trajectory = [root]
        curr = root
        while curr.children:
            # En inferencia final, la política estándar es seleccionar el hijo más visitado (más robusto)
            curr = max(curr.children, key=lambda c: (c.visit_count, c.value))
            trajectory.append(curr)
        return trajectory

    def _format_solution(self, prompt: str, trajectory: List[MCTSNode]) -> str:
        """Formatea la trayectoria seleccionada en el estándar nativo Qwen3 <think>...</think>."""
        steps = [node.state_text for node in trajectory[1:]]  # Omitir root_prompt
        inner_thoughts = "\n".join(f"Paso {i+1}: {step}" for i, step in enumerate(steps))
        return f"<think>\n{inner_thoughts}\n</think>\nPor lo tanto, la solución queda demostrada."
