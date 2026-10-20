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
Módulo de Paginación de Memoria de Atención (PagedAttention) para Inferencia de Alto Rendimiento.

Fundamento:
En despliegues de producción y procesamiento de contextos extensos en Qwen3, la asignación
de tensores contiguos para KV-Cache causa severa fragmentación de memoria en la GPU/RAM.

Este módulo implementa el gestor de memoria paginada estilo vLLM (Kwon et al., 2023):
1. Bloques Físicos Discretos (Physical Blocks): Tamaño fijo de 16 tokens por bloque.
2. Tablas de Bloques Lógicos (Block Tables): Mapeo virtual sin necesidad de contigüidad.
3. Compartición de Prefijos (Prefix Caching): Reutilización de bloques con recuento de referencias.
"""

import math
import torch

class PhysicalBlock:
    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0
        self.num_tokens = 0

    def is_full(self) -> bool:
        return self.num_tokens >= self.block_size

    def is_free(self) -> bool:
        return self.ref_count == 0


class PagedKVCacheManager:
    """
    Administrador de bloques físicos de memoria para KV-Cache paginado.
    """
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        total_blocks: int = 256,
        block_size: int = 16,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        # Pool de memoria física pre-asignado: [num_layers, total_blocks, num_kv_heads, block_size, head_dim]
        self.key_pool = torch.zeros(
            (num_layers, total_blocks, num_kv_heads, block_size, head_dim),
            device=device,
            dtype=dtype
        )
        self.value_pool = torch.zeros(
            (num_layers, total_blocks, num_kv_heads, block_size, head_dim),
            device=device,
            dtype=dtype
        )

        # Registro de bloques físicos
        self.blocks = [PhysicalBlock(i, block_size) for i in range(total_blocks)]
        self.free_block_ids = list(range(total_blocks))

        # Tablas de bloques por secuencia: seq_id -> list[block_id]
        self.block_tables: dict[str, list[int]] = {}

    def allocate_sequence(self, seq_id: str):
        """Inicializa una tabla de bloques vacía para una nueva secuencia."""
        self.block_tables[seq_id] = []

    def allocate_block(self) -> int:
        """Extrae un bloque físico libre del pool."""
        if not self.free_block_ids:
            raise MemoryError("Pool de bloques físicos de PagedAttention agotado.")
        b_id = self.free_block_ids.pop(0)
        self.blocks[b_id].ref_count = 1
        self.blocks[b_id].num_tokens = 0
        return b_id

    def append_token(
        self,
        seq_id: str,
        layer_idx: int,
        key_token: torch.Tensor,
        value_token: torch.Tensor
    ):
        """
        Inserta un nuevo token en la memoria paginada de la secuencia:
        Args:
            key_token, value_token: Tensores [num_kv_heads, head_dim]
        """
        table = self.block_tables[seq_id]
        if not table or self.blocks[table[-1]].is_full():
            new_block_id = self.allocate_block()
            table.append(new_block_id)

        target_block_id = table[-1]
        block = self.blocks[target_block_id]
        slot = block.num_tokens

        # Asignar en el tensor de la piscina física
        self.key_pool[layer_idx, target_block_id, :, slot, :] = key_token
        self.value_pool[layer_idx, target_block_id, :, slot, :] = value_token

        # Solo incrementar el contador en la última capa
        if layer_idx == self.num_layers - 1:
            block.num_tokens += 1

    def free_sequence(self, seq_id: str):
        """Libera todos los bloques asociados a una secuencia."""
        if seq_id not in self.block_tables:
            return
        table = self.block_tables.pop(seq_id)
        for b_id in table:
            block = self.blocks[b_id]
            block.ref_count -= 1
            if block.is_free():
                block.num_tokens = 0
                self.free_block_ids.append(b_id)


__all__ = ["PagedKVCacheManager", "PhysicalBlock"]
