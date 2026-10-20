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
Continuous Batching & Iteration-Level Request Scheduler
========================================================
Planificador de lotes continuos a nivel de iteración (vLLM / TGI style) para Qwen3.

Capacidades:
1. Iteration-Level Scheduling: Permite que nuevas solicitudes se unan al lote
   de generación inmediatamente en el siguiente token, sin esperar a que las
   solicitudes existentes terminen de decodificar.
2. Prefill / Decode Interleaving: Coordina la ejecución simultánea de prefill y
   pasos de decodificación token por token.
3. Gestión del ciclo de vida de peticiones (PENDING, PREFILLING, DECODING, FINISHED).
"""

import time
from enum import Enum
from typing import Dict, List, Optional, Any
import torch


class RequestState(Enum):
    PENDING = 0
    PREFILLING = 1
    DECODING = 2
    FINISHED = 3


class SequenceRequest:
    """Representa una petición individual dentro de la cola del servidor."""
    def __init__(
        self,
        request_id: str,
        prompt_tokens: List[int],
        max_new_tokens: int = 128,
        eos_token_id: int = 151643
    ):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens: List[int] = []
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.state = RequestState.PENDING
        self.created_at = time.time()

    @property
    def is_finished(self) -> bool:
        return self.state == RequestState.FINISHED

    @property
    def current_tokens(self) -> List[int]:
        return self.prompt_tokens + self.output_tokens


class ContinuousBatchingScheduler:
    """
    Planificador de generación concurrente que maximiza el rendimiento (tokens/segundo)
    dinamizando el tamaño del lote en cada paso del modelo.
    """
    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        self.waiting_queue: List[SequenceRequest] = []
        self.running_batch: List[SequenceRequest] = []
        self.finished_requests: Dict[str, SequenceRequest] = {}

    def add_request(self, request: SequenceRequest):
        """Añade una nueva solicitud a la cola de espera."""
        self.waiting_queue.append(request)

    def schedule(self) -> List[SequenceRequest]:
        """
        Planifica el lote de solicitudes activas para la iteración actual.
        Si hay espacio disponible, promueve solicitudes de waiting_queue.
        """
        # Limpiar solicitudes terminadas
        active = []
        for req in self.running_batch:
            if req.is_finished:
                self.finished_requests[req.request_id] = req
            else:
                active.append(req)
        self.running_batch = active

        # Admitir nuevas solicitudes si hay capacidad
        while len(self.running_batch) < self.max_batch_size and self.waiting_queue:
            req = self.waiting_queue.pop(0)
            req.state = RequestState.PREFILLING
            self.running_batch.append(req)

        return self.running_batch

    def step(self, sample_fn: Any) -> Dict[str, Optional[int]]:
        """
        Ejecuta un ciclo de decodificación para todas las solicitudes activas.
        
        Args:
            sample_fn: Función (request) -> next_token_id generada por el modelo
            
        Returns:
            step_outputs: Diccionario {request_id: next_token_id}
        """
        active_requests = self.schedule()
        step_outputs = {}

        for req in active_requests:
            if req.state == RequestState.PREFILLING:
                # Transición de prefill a decode
                next_tok = sample_fn(req)
                req.output_tokens.append(next_tok)
                req.state = RequestState.DECODING
                step_outputs[req.request_id] = next_tok
            elif req.state == RequestState.DECODING:
                next_tok = sample_fn(req)
                req.output_tokens.append(next_tok)
                step_outputs[req.request_id] = next_tok

            # Comprobar condiciones de parada
            if next_tok == req.eos_token_id or len(req.output_tokens) >= req.max_new_tokens:
                req.state = RequestState.FINISHED

        return step_outputs
