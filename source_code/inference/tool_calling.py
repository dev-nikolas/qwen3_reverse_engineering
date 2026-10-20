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
Motor de Llamada a Herramientas (Tool / Function Calling Engine) para Qwen3.

Fundamento Forense:
En el análisis del vocabulario y tokens especiales de Qwen3 se identificaron las
etiquetas estructuradas de ejecución de funciones:
  - <tool_call> ... </tool_call>
  - <tool_response> ... </tool_response>

Este motor:
1. Serializa descripciones de funciones (JSON Schema) en el prompt de sistema canónico de Qwen.
2. Parsea en tiempo de inferencia las emisiones de llamadas a funciones en JSON.
3. Formatea la respuesta de retorno para el siguiente turno de razonamiento.
"""

import json
import re

class Qwen3ToolCallingEngine:
    """
    Controlador de ejecución y serialización de herramientas para Qwen3.
    """
    TOOL_CALL_START = "<tool_call>"
    TOOL_CALL_END = "</tool_call>"
    TOOL_RESP_START = "<tool_response>"
    TOOL_RESP_END = "</tool_response>"

    def __init__(self, tools_definitions: list[dict] | None = None):
        self.tools = tools_definitions or []

    def format_system_prompt_with_tools(self, base_system_message: str = "Eres un asistente servicial.") -> str:
        """
        Formatea el mensaje de sistema inyectando el catálogo de herramientas disponibles.
        """
        if not self.tools:
            return base_system_message

        tools_json = json.dumps(self.tools, indent=2, ensure_ascii=False)
        system_prompt = (
            f"{base_system_message}\n\n"
            f"# Herramientas Disponibles\n"
            f"Tienes acceso a las siguientes funciones:\n```json\n{tools_json}\n```\n\n"
            f"Para invocar una herramienta, emite exactamente:\n"
            f"{self.TOOL_CALL_START}\n"
            f'{{"name": "nombre_de_funcion", "arguments": {{"param": "valor"}}}}\n'
            f"{self.TOOL_CALL_END}"
        )
        return system_prompt

    def parse_tool_calls(self, text: str) -> list[dict]:
        """
        Extrae y valida todas las llamadas a herramientas presentes en el texto generado.
        
        Returns:
            Lista de diccionarios [{'name': str, 'arguments': dict, 'raw': str}]
        """
        tool_calls = []
        pattern = re.compile(rf"{re.escape(self.TOOL_CALL_START)}(.*?){re.escape(self.TOOL_CALL_END)}", re.DOTALL)
        matches = pattern.findall(text)

        for raw_payload in matches:
            cleaned = raw_payload.strip()
            # Limpiar bloques de markdown si el modelo los emitió adentro
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "name" in parsed:
                    tool_calls.append({
                        "name": parsed["name"],
                        "arguments": parsed.get("arguments", {}),
                        "raw": raw_payload
                    })
            except json.JSONDecodeError:
                # Intento de extracción heurística si el JSON quedó truncado
                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', cleaned)
                if name_match:
                    tool_calls.append({
                        "name": name_match.group(1),
                        "arguments": {},
                        "raw": raw_payload,
                        "error": "JSON malformado"
                    })

        return tool_calls

    def format_tool_response(self, tool_name: str, response_payload: dict | str) -> str:
        """
        Formatea la respuesta devuelta por la herramienta para inyectarla de regreso al modelo.
        """
        if isinstance(response_payload, dict):
            resp_str = json.dumps(response_payload, ensure_ascii=False)
        else:
            resp_str = str(response_payload)

        return f"\n{self.TOOL_RESP_START}\n{resp_str}\n{self.TOOL_RESP_END}\n"


__all__ = ["Qwen3ToolCallingEngine"]
