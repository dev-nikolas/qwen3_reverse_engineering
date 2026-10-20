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
Motor de Relleno en el Medio (Fill-In-The-Middle / FIM Engine) para Qwen3.

Fundamento Forense:
En el análisis forense del vocabulario de 151,936 tokens de Qwen3 se identificaron
los tokens nativos dedicados a autocompletado y edición de código:
  - <|fim_prefix|> (ID 151659)
  - <|fim_middle|> (ID 151660)
  - <|fim_suffix|> (ID 151661)
  - <|repo_name|>  (ID 151664)
  - <|file_sep|>   (ID 151665)

Este motor formatea las solicitudes de inserción de código con el estándar oficial de Qwen,
gestiona la generación del fragmento central y reconstruye el archivo completo.
"""

import torch

class FIMEngine:
    """
    Controlador y formateador de completado de código Fill-In-The-Middle.
    """
    FIM_PREFIX = "<|fim_prefix|>"
    FIM_MIDDLE = "<|fim_middle|>"
    FIM_SUFFIX = "<|fim_suffix|>"
    REPO_NAME = "<|repo_name|>"
    FILE_SEP = "<|file_sep|>"
    ENDOFTEXT = "<|endoftext|>"

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def format_prompt(
        self,
        prefix: str,
        suffix: str,
        repo_name: str | None = None,
        file_path: str | None = None
    ) -> str:
        """
        Ensambla el prompt canónico FIM:
        [<|repo_name|>repo][<|file_sep|>file]<|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>
        """
        prompt = ""
        if repo_name:
            prompt += f"{self.REPO_NAME}{repo_name}"
        if file_path:
            prompt += f"{self.FILE_SEP}{file_path}"
            
        prompt += f"{self.FIM_PREFIX}{prefix}{self.FIM_SUFFIX}{suffix}{self.FIM_MIDDLE}"
        return prompt

    def complete(
        self,
        prefix: str,
        suffix: str,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        repo_name: str | None = None,
        file_path: str | None = None
    ) -> dict[str, str]:
        """
        Ejecuta la inferencia FIM para predecir el código faltante en el medio.
        
        Returns:
            Diccionario con:
            - 'middle': el código generado entre prefix y suffix
            - 'full_code': prefix + middle + suffix
        """
        prompt = self.format_prompt(prefix, suffix, repo_name=repo_name, file_path=file_path)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                eos_token_id=self.tokenizer.encode(self.ENDOFTEXT)[-1]
            )

        # Extraer únicamente los tokens nuevos generados
        generated_tokens = output_ids[0][input_ids.shape[1]:]
        middle_code = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {
            "middle": middle_code,
            "full_code": prefix + middle_code + suffix
        }


__all__ = ["FIMEngine"]
