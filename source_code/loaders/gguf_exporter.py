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
GGUF Binary Exporter for Qwen3-1.7B
====================================
Exportador binario puro en Python/PyTorch para serializar pesos y metadatos
de Qwen3 al formato estándar GGUF v3, compatible con llama.cpp, Ollama,
LM Studio y vLLM.

Soporta:
- Mapeo canónico de nombres HuggingFace/Qwen3 a GGML tensor nomenclature.
- Metadatos estandarizados de arquitectura (RoPE theta 1,000,000, GQA 16:8, context 32k/131k).
- Tipos de datos: F32, F16 y cuantización GGML Q8_0 (bloques de 32 valores con escala fp16).
- Alineación estricta de 32 bytes para carga eficiente con mmap().
"""

import os
import struct
import io
from typing import Dict, Any, Optional
import torch

# GGUF Constants
GGUF_MAGIC = 0x46554747  # b"GGUF" in little endian
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# GGUF Value Types
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

# GGML Tensor Types
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q8_0 = 8

# Mapeo de nombres de tensores Qwen3 a GGML
HF_TO_GGML_MAPPING = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}


def map_qwen3_tensor_name(hf_name: str) -> str:
    """Mapea nombres de tensores HuggingFace a nombres estándar GGML."""
    if hf_name in HF_TO_GGML_MAPPING:
        return HF_TO_GGML_MAPPING[hf_name]
    
    parts = hf_name.split(".")
    if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
        layer_idx = parts[2]
        sub = ".".join(parts[3:])
        
        layer_map = {
            "self_attn.q_proj.weight": f"blk.{layer_idx}.attn_q.weight",
            "self_attn.k_proj.weight": f"blk.{layer_idx}.attn_k.weight",
            "self_attn.v_proj.weight": f"blk.{layer_idx}.attn_v.weight",
            "self_attn.o_proj.weight": f"blk.{layer_idx}.attn_output.weight",
            "mlp.gate_proj.weight": f"blk.{layer_idx}.ffn_gate.weight",
            "mlp.up_proj.weight": f"blk.{layer_idx}.ffn_up.weight",
            "mlp.down_proj.weight": f"blk.{layer_idx}.ffn_down.weight",
            "input_layernorm.weight": f"blk.{layer_idx}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{layer_idx}.ffn_norm.weight",
        }
        if sub in layer_map:
            return layer_map[sub]
            
    return hf_name


def quantize_to_q8_0(tensor: torch.Tensor) -> bytes:
    """
    Cuantiza un tensor float32/float16 a formato GGML Q8_0.
    Estructura por bloque de 32 floats:
    - delta: float16 (2 bytes)
    - qs: int8[32] (32 bytes)
    Total: 34 bytes por cada 32 elementos (reducción a ~8.5 bits/peso).
    """
    flat = tensor.detach().to(torch.float32).flatten()
    n_elements = flat.numel()
    
    # Alinear al múltiplo de 32 más cercano
    remainder = n_elements % 32
    if remainder != 0:
        pad_size = 32 - remainder
        flat = torch.nn.functional.pad(flat, (0, pad_size))
        
    blocks = flat.view(-1, 32)
    max_vals = torch.amax(torch.abs(blocks), dim=1, keepdim=True)
    # Evitar división por cero
    scales = torch.clamp(max_vals / 127.0, min=1e-8)
    
    # Cuantizar a int8 [-127, 127]
    quantized_blocks = torch.clamp(torch.round(blocks / scales), -127, 127).to(torch.int8)
    
    # Serializar bloques
    out = bytearray()
    scales_fp16 = scales.squeeze(1).to(torch.float16).cpu().numpy()
    qs_int8 = quantized_blocks.cpu().numpy()
    
    for i in range(len(scales_fp16)):
        out.extend(scales_fp16[i].tobytes())
        out.extend(qs_int8[i].tobytes())
        
    return bytes(out)


class Qwen3GGUFExporter:
    """
    Serializador de arquitectura y pesos de Qwen3 al formato binario GGUF v3.
    """
    
    def __init__(self, alignment: int = DEFAULT_ALIGNMENT):
        self.alignment = alignment
        
    def _write_string(self, buf: io.BytesIO, s: str):
        encoded = s.encode("utf-8")
        buf.write(struct.pack("<Q", len(encoded)))
        buf.write(encoded)
        
    def _write_kv(self, buf: io.BytesIO, key: str, value: Any, val_type: int):
        self._write_string(buf, key)
        buf.write(struct.pack("<I", val_type))
        
        if val_type == GGUF_TYPE_UINT32:
            buf.write(struct.pack("<I", int(value)))
        elif val_type == GGUF_TYPE_INT32:
            buf.write(struct.pack("<i", int(value)))
        elif val_type == GGUF_TYPE_UINT64:
            buf.write(struct.pack("<Q", int(value)))
        elif val_type == GGUF_TYPE_INT64:
            buf.write(struct.pack("<q", int(value)))
        elif val_type == GGUF_TYPE_FLOAT32:
            buf.write(struct.pack("<f", float(value)))
        elif val_type == GGUF_TYPE_BOOL:
            buf.write(struct.pack("<B", 1 if value else 0))
        elif val_type == GGUF_TYPE_STRING:
            self._write_string(buf, str(value))
        elif val_type == GGUF_TYPE_ARRAY:
            elem_type, arr = value
            buf.write(struct.pack("<I", elem_type))
            buf.write(struct.pack("<Q", len(arr)))
            for item in arr:
                if elem_type == GGUF_TYPE_STRING:
                    self._write_string(buf, str(item))
                elif elem_type == GGUF_TYPE_FLOAT32:
                    buf.write(struct.pack("<f", float(item)))
                elif elem_type == GGUF_TYPE_INT32:
                    buf.write(struct.pack("<i", int(item)))
                else:
                    raise ValueError(f"Tipo de array no soportado: {elem_type}")
        else:
            raise ValueError(f"Tipo de valor GGUF no soportado: {val_type}")

    def get_default_qwen3_metadata(self) -> Dict[str, tuple]:
        """Metadatos canónicos de la arquitectura Qwen3-1.7B según ingeniería inversa."""
        return {
            "general.architecture": ("qwen2", GGUF_TYPE_STRING),
            "general.name": ("Qwen3-1.7B", GGUF_TYPE_STRING),
            "qwen2.context_length": (32768, GGUF_TYPE_UINT32),
            "qwen2.embedding_length": (2048, GGUF_TYPE_UINT32),
            "qwen2.block_count": (28, GGUF_TYPE_UINT32),
            "qwen2.feed_forward_length": (6144, GGUF_TYPE_UINT32),
            "qwen2.attention.head_count": (16, GGUF_TYPE_UINT32),
            "qwen2.attention.head_count_kv": (8, GGUF_TYPE_UINT32),
            "qwen2.rope.freq_base": (1000000.0, GGUF_TYPE_FLOAT32),
            "qwen2.attention.layer_norm_rms_epsilon": (1e-6, GGUF_TYPE_FLOAT32),
        }

    def export(
        self,
        tensors: Dict[str, torch.Tensor],
        output_path: str,
        quantization: str = "f16",
        extra_metadata: Optional[Dict[str, tuple]] = None
    ):
        """
        Exporta el diccionario de tensores a un archivo binario GGUF.
        
        Args:
            tensors: Diccionario {nombre: tensor_torch}
            output_path: Ruta destino del archivo .gguf
            quantization: Formato de pesos: 'f32', 'f16' o 'q8_0'
            extra_metadata: Diccionario opcional de metadatos adicionales
        """
        metadata = self.get_default_qwen3_metadata()
        if extra_metadata:
            metadata.update(extra_metadata)
            
        quant_lower = quantization.lower()
        if quant_lower not in ["f32", "f16", "q8_0"]:
            raise ValueError(f"Cuantización no soportada: {quantization}")

        header_buf = io.BytesIO()
        # 1. Cabecera GGUF
        header_buf.write(struct.pack("<I", GGUF_MAGIC))
        header_buf.write(struct.pack("<I", GGUF_VERSION))
        header_buf.write(struct.pack("<Q", len(tensors)))
        header_buf.write(struct.pack("<Q", len(metadata)))

        # 2. Escribir Metadatos
        for key, (val, val_type) in metadata.items():
            self._write_kv(header_buf, key, val, val_type)

        # 3. Preparar información de tensores y datos binarios
        tensor_infos = []
        tensor_data_buffers = []
        current_offset = 0

        for hf_name, tensor in tensors.items():
            ggml_name = map_qwen3_tensor_name(hf_name)
            shape = list(tensor.shape)
            # En GGML las dimensiones se ordenan de la más interna a la más externa
            dims_ggml = shape[::-1]
            n_dims = len(dims_ggml)

            # Selección de tipo y empaquetado binario
            if quant_lower == "f32":
                raw_bytes = tensor.detach().to(torch.float32).cpu().numpy().tobytes()
                ggml_type = GGML_TYPE_F32
            elif quant_lower == "f16":
                raw_bytes = tensor.detach().to(torch.float16).cpu().numpy().tobytes()
                ggml_type = GGML_TYPE_F16
            elif quant_lower == "q8_0":
                # No cuantizamos normas ni vectores de 1D en Q8_0, solo matrices
                if n_dims > 1 and "norm" not in ggml_name:
                    raw_bytes = quantize_to_q8_0(tensor)
                    ggml_type = GGML_TYPE_Q8_0
                else:
                    raw_bytes = tensor.detach().to(torch.float16).cpu().numpy().tobytes()
                    ggml_type = GGML_TYPE_F16
            else:
                raw_bytes = tensor.detach().to(torch.float16).cpu().numpy().tobytes()
                ggml_type = GGML_TYPE_F16

            # Alinear el offset relativo
            pad = (self.alignment - (current_offset % self.alignment)) % self.alignment
            current_offset += pad
            assigned_offset = current_offset

            tensor_infos.append({
                "name": ggml_name,
                "n_dims": n_dims,
                "dims": dims_ggml,
                "type": ggml_type,
                "offset": assigned_offset,
            })

            tensor_data_buffers.append((pad, raw_bytes))
            current_offset += len(raw_bytes)

        # 4. Escribir tabla de metadatos de tensores
        for info in tensor_infos:
            self._write_string(header_buf, info["name"])
            header_buf.write(struct.pack("<I", info["n_dims"]))
            for d in info["dims"]:
                header_buf.write(struct.pack("<Q", d))
            header_buf.write(struct.pack("<I", info["type"]))
            header_buf.write(struct.pack("<Q", info["offset"]))

        # 5. Alinear encabezado completo con el inicio del bloque de datos
        header_bytes = header_buf.getvalue()
        header_len = len(header_bytes)
        final_pad = (self.alignment - (header_len % self.alignment)) % self.alignment

        # 6. Escribir a disco
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(header_bytes)
            if final_pad > 0:
                f.write(b"\x00" * final_pad)
            for pad_bytes, data_bytes in tensor_data_buffers:
                if pad_bytes > 0:
                    f.write(b"\x00" * pad_bytes)
                f.write(data_bytes)

        return {
            "output_path": output_path,
            "tensors_exported": len(tensors),
            "quantization": quant_lower,
            "file_size_bytes": os.path.getsize(output_path),
        }
