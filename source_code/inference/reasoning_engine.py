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
Motor de Razonamiento Reflexivo (Chain-of-Thought Reasoning Engine) para Qwen3.

Fundamento Forense:
En el análisis del vocabulario de Qwen3 se identificaron los tokens delimitadores de pensamiento:
  - <think>  (ID 151667)
  - </think> (ID 151668)

Este motor administra el ciclo de vida del razonamiento deliberativo:
1. Control de Presupuesto: Limita o fuerza la duración del pensamiento (`max_thinking_tokens`).
2. Detección de Rectificación (Backtracking): Identifica cuando el modelo detecta un error
   en su propio razonamiento ("Espera...", "Sin embargo...", "Reconsiderando...").
3. Transición Suave: Si el presupuesto expira, inyecta el cierre `</think>\n` y fuerza la
   emisión de la respuesta final.
4. Extracción Estructurada: Separa la traza de pensamiento privada de la respuesta pública.
"""

import re
import torch
from .sampling import sample_next_token

class Qwen3ReasoningEngine:
    """
    Controlador de razonamiento deliberativo y presupuestación de cadena de pensamiento.
    """
    THINK_START = "<think>"
    THINK_END = "</think>"
    
    # Patrones de reflexión y rectificación cognitiva
    REFLECTION_CUES = [
        r"(?i)\bespera\b",
        r"(?i)\breconsiderando\b",
        r"(?i)\bsin embargo\b",
        r"(?i)\bwait\b",
        r"(?i)\blet me rethink\b",
        r"(?i)\balternatively\b",
        r"(?i)\bactually\b",
        r"(?i)\bpero\b"
    ]

    def __init__(
        self,
        model=None,
        tokenizer=None,
        max_thinking_tokens: int = 512,
        device: str = "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.device = device
        self.compiled_cues = [re.compile(p) for p in self.REFLECTION_CUES]

    def count_backtracking_events(self, text: str) -> int:
        """Cuenta los momentos de rectificación o giro reflexivo en la traza."""
        count = 0
        for pattern in self.compiled_cues:
            count += len(pattern.findall(text))
        return count

    def generate_with_reasoning(
        self,
        prompt: str,
        max_answer_tokens: int = 256,
        temperature_thinking: float = 0.6,
        temperature_answer: float = 0.3,
        force_thinking: bool = True
    ) -> dict[str, any]:
        """
        Ejecuta la generación deliberativa en dos fases acopladas.
        
        Returns:
            {
                'thought': str,
                'answer': str,
                'total_text': str,
                'thinking_tokens': int,
                'backtracking_events': int,
                'budget_exhausted': bool
            }
        """
        # Preparar prompt asegurando el inicio de <think> si está habilitado
        working_prompt = prompt.strip()
        if force_thinking and not working_prompt.endswith(self.THINK_START):
            working_prompt += f"\n{self.THINK_START}\n"

        input_ids = self.tokenizer.encode(working_prompt, return_tensors="pt").to(self.device)
        generated_ids = input_ids[0].tolist()
        
        in_thinking_phase = True
        thinking_token_count = 0
        budget_exhausted = False
        answer_token_count = 0

        # Identificar IDs de tokens de control si el tokenizer los soporta
        think_end_id = None
        try:
            think_end_id = self.tokenizer.encode(self.THINK_END, add_special_tokens=False)[-1]
        except Exception:
            pass

        # Bucle autoregresivo
        for _ in range(self.max_thinking_tokens + max_answer_tokens):
            curr_input = torch.tensor([generated_ids], device=self.device)
            with torch.no_grad():
                logits = self.model(curr_input)
                next_logits = logits[:, -1, :]

            current_temp = temperature_thinking if in_thinking_phase else temperature_answer
            next_token = sample_next_token(next_logits, temperature=current_temp)

            # Manejo de fase de pensamiento
            if in_thinking_phase:
                thinking_token_count += 1
                token_str = self.tokenizer.decode([next_token])

                # Detección de cierre natural de pensamiento
                if (think_end_id is not None and next_token == think_end_id) or (self.THINK_END in token_str):
                    in_thinking_phase = False
                    generated_ids.append(next_token)
                    continue

                # Control de agotamiento de presupuesto
                if thinking_token_count >= self.max_thinking_tokens:
                    budget_exhausted = True
                    in_thinking_phase = False
                    # Forzar inyección de cierre de pensamiento
                    closure_tokens = self.tokenizer.encode(f"\n{self.THINK_END}\n", add_special_tokens=False)
                    generated_ids.extend(closure_tokens)
                    continue

                generated_ids.append(next_token)
            else:
                # Fase de respuesta final
                answer_token_count += 1
                token_str = self.tokenizer.decode([next_token])
                generated_ids.append(next_token)

                # Si emite fin de texto
                if "<|endoftext|>" in token_str or "<|im_end|>" in token_str or answer_token_count >= max_answer_tokens:
                    break

        full_output = self.tokenizer.decode(generated_ids[input_ids.shape[1]:])
        
        # Desglosar pensamiento y respuesta mediante regex
        thought_match = re.search(r"<think>(.*?)</think>", full_output, re.DOTALL)
        if thought_match:
            thought_text = thought_match.group(1).strip()
            answer_text = full_output.split("</think>")[-1].strip()
        else:
            thought_text = full_output.strip()
            answer_text = ""

        backtracking_events = self.count_backtracking_events(thought_text)

        return {
            "thought": thought_text,
            "answer": answer_text,
            "full_output": full_output,
            "thinking_tokens": thinking_token_count,
            "backtracking_events": backtracking_events,
            "budget_exhausted": budget_exhausted
        }


__all__ = ["Qwen3ReasoningEngine"]
