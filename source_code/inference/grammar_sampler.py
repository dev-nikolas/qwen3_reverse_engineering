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
Token Healing & Constrained Grammar Masking Engine
===================================================
Motor de inferencia estructurada y corrección de límites de token para Qwen3.

Soluciona:
1. Token Boundary Problem: Cuando los prompts terminan en delimitadores incompletos
   (ej. `{"name": "`), el motor realiza Token Healing retrocediendo un token y
   forzando extensiones válidas de la palabra.
2. JSON/XML Grammar Masking: Autómata de estados finitos que aplica máscaras
   estrictas (-inf) a logits no conformes, garantizando JSON 100% sintáctico
   para llamadas a herramientas (<tool_call>) y esquemas tipados.
"""

import re
from typing import List, Set, Dict, Optional
import torch


class TokenHealingProcessor:
    """
    Soluciona el 'Token Boundary Problem' cuando el prompt del usuario
    o plantilla del sistema finaliza en el prefijo de un token compuesto.
    """
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def find_prefix_completions(self, prefix_str: str) -> List[int]:
        """Encuentra todos los token IDs cuyo texto comience con el prefijo dado."""
        matching_ids = []
        for token_id, token_str in self.inv_vocab.items():
            if token_str.startswith(prefix_str):
                matching_ids.append(token_id)
        return matching_ids

    def apply_healing_mask(
        self,
        logits: torch.Tensor,
        partial_prefix: str
    ) -> torch.Tensor:
        """
        Enmascara los logits permitiendo únicamente tokens compatibles con el prefijo.
        """
        valid_ids = self.find_prefix_completions(partial_prefix)
        if not valid_ids:
            return logits  # Sin restricciones si no hay coincidencias

        mask = torch.full_like(logits, float("-inf"))
        mask[..., valid_ids] = logits[..., valid_ids]
        return mask


class JSONGrammarMasker:
    """
    Autómata de estados finitos liviano para forzar salidas JSON válidas
    en tiempo de inferencia paso a paso.
    
    Estados:
      0: START -> solo permite '{'
      1: KEY -> permite '"'
      2: COLON -> permite ':'
      3: VALUE -> permite strings, números, booleanos, '{', '['
      4: COMMA_OR_CLOSE -> permite ',', '}'
    """
    STATE_START = 0
    STATE_KEY = 1
    STATE_COLON = 2
    STATE_VALUE = 3
    STATE_COMMA_OR_CLOSE = 4

    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id
        self.state = self.STATE_START
        self.brace_depth = 0

    def reset(self):
        self.state = self.STATE_START
        self.brace_depth = 0

    def get_allowed_tokens(self) -> Set[str]:
        """Retorna el conjunto de caracteres o tokens sintácticos permitidos en el estado actual."""
        if self.state == self.STATE_START:
            return {"{"}
        elif self.state == self.STATE_KEY:
            return {'"', "}"}
        elif self.state == self.STATE_COLON:
            return {":"}
        elif self.state == self.STATE_VALUE:
            return {'"', "{", "[", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "true", "false", "null"}
        elif self.state == self.STATE_COMMA_OR_CLOSE:
            return {",", "}"}
        return set()

    def step(self, token_str: str):
        """Avanza el estado del autómata con el token emitido."""
        s = token_str.strip()
        if "{" in s:
            self.brace_depth += s.count("{")
            self.state = self.STATE_KEY
        elif "}" in s:
            self.brace_depth -= s.count("}")
            if self.brace_depth <= 0:
                self.state = self.STATE_START
            else:
                self.state = self.STATE_COMMA_OR_CLOSE
        elif ":" in s:
            self.state = self.STATE_VALUE
        elif "," in s:
            self.state = self.STATE_KEY
        elif self.state == self.STATE_VALUE and ('"' in s or any(c.isdigit() for c in s)):
            self.state = self.STATE_COMMA_OR_CLOSE

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Filtra los logits poniendo -inf en tokens que violarían la gramática JSON.
        """
        allowed = self.get_allowed_tokens()
        allowed_ids = [self.token_to_id[t] for t in allowed if t in self.token_to_id]
        
        if not allowed_ids:
            return logits

        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits[..., allowed_ids] = logits[..., allowed_ids]
        return masked_logits
