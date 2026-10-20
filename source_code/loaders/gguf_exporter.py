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
import math
from typing import Dict, Any, Optional, List
import torch  # type: ignore

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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, alignment: int = DEFAULT_ALIGNMENT):
        if isinstance(config, int):
            self.alignment = config
            self.config = None
        else:
            self.config = config
            self.alignment = alignment if isinstance(alignment, int) else DEFAULT_ALIGNMENT
        
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

        # 3. Primer pase: calcular metadatos y offsets exactos de los tensores
        tensor_infos = []
        current_offset = 0

        for hf_name, tensor in tensors.items():
            ggml_name = map_qwen3_tensor_name(hf_name)
            shape = list(tensor.shape)
            dims_ggml = shape[::-1]
            n_dims = len(dims_ggml)
            numel = math.prod(shape)

            if quant_lower == "f32":
                raw_len = numel * 4
                ggml_type = GGML_TYPE_F32
            elif quant_lower == "f16":
                raw_len = numel * 2
                ggml_type = GGML_TYPE_F16
            elif quant_lower == "q8_0":
                if n_dims > 1 and "norm" not in ggml_name:
                    pad_numel = (32 - (numel % 32)) % 32
                    raw_len = ((numel + pad_numel) // 32) * 34
                    ggml_type = GGML_TYPE_Q8_0
                else:
                    raw_len = numel * 2
                    ggml_type = GGML_TYPE_F16
            else:
                raw_len = numel * 2
                ggml_type = GGML_TYPE_F16

            pad = (self.alignment - (current_offset % self.alignment)) % self.alignment
            current_offset += pad
            assigned_offset = current_offset

            tensor_infos.append({
                "hf_name": hf_name,
                "name": ggml_name,
                "n_dims": n_dims,
                "dims": dims_ggml,
                "type": ggml_type,
                "offset": assigned_offset,
                "pad": pad
            })
            current_offset += raw_len

        # 4. Escribir tabla de metadatos de tensores en el header
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

        # 6. Segundo pase: Escritura en Streaming directa a disco tensor por tensor (O(1) memoria)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(header_bytes)
            if final_pad > 0:
                f.write(b"\x00" * final_pad)

            for info in tensor_infos:
                if info["pad"] > 0:
                    f.write(b"\x00" * info["pad"])

                tensor = tensors[info["hf_name"]]
                if info["type"] == GGML_TYPE_F32:
                    raw_bytes = tensor.detach().to(torch.float32).cpu().numpy().tobytes()
                elif info["type"] == GGML_TYPE_Q8_0:
                    raw_bytes = quantize_to_q8_0(tensor)
                else:  # F16
                    raw_bytes = tensor.detach().to(torch.float16).cpu().numpy().tobytes()

                f.write(raw_bytes)

        return {
            "output_path": output_path,
            "tensors_exported": len(tensors),
            "quantization": quant_lower,
            "file_size_bytes": os.path.getsize(output_path),
        }

    def export_safetensors_files(
        self,
        safetensor_paths: List[str],
        output_path: str,
        quantization: str = "f16"
    ) -> Dict[str, Any]:
        """
        Exportación directa en streaming desde archivos Safetensors a GGUF v3.
        Consumo de memoria O(1) (< 100 MB de RAM) ya que lee y serializa un tensor a la vez.
        """
        from safetensors import safe_open  # type: ignore
        import gc

        quant_lower = quantization.lower()
        if quant_lower not in ["f32", "f16", "q8_0"]:
            raise ValueError(f"Cuantización no soportada: {quantization}")

        metadata = self.get_default_qwen3_metadata()

        # Mapear cada tensor a su archivo correspondiente sin cargar tensores a RAM
        tensor_to_file = {}
        tensor_shapes = {}
        for s_path in safetensor_paths:
            with safe_open(s_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensor_to_file[k] = s_path
                    tensor_shapes[k] = f.get_slice(k).get_shape()

        header_buf = io.BytesIO()
        header_buf.write(struct.pack("<I", GGUF_MAGIC))
        header_buf.write(struct.pack("<I", GGUF_VERSION))
        header_buf.write(struct.pack("<Q", len(tensor_to_file)))
        header_buf.write(struct.pack("<Q", len(metadata)))

        for key, (val, val_type) in metadata.items():
            self._write_kv(header_buf, key, val, val_type)

        tensor_infos = []
        current_offset = 0

        for hf_name, shape in tensor_to_file.items():
            ggml_name = map_qwen3_tensor_name(hf_name)
            t_shape = tensor_shapes[hf_name]
            dims_ggml = t_shape[::-1]
            n_dims = len(dims_ggml)
            numel = math.prod(t_shape)

            if quant_lower == "f32":
                raw_len = numel * 4
                ggml_type = GGML_TYPE_F32
            elif quant_lower == "f16":
                raw_len = numel * 2
                ggml_type = GGML_TYPE_F16
            elif quant_lower == "q8_0":
                if n_dims > 1 and "norm" not in ggml_name:
                    pad_numel = (32 - (numel % 32)) % 32
                    raw_len = ((numel + pad_numel) // 32) * 34
                    ggml_type = GGML_TYPE_Q8_0
                else:
                    raw_len = numel * 2
                    ggml_type = GGML_TYPE_F16
            else:
                raw_len = numel * 2
                ggml_type = GGML_TYPE_F16

            pad = (self.alignment - (current_offset % self.alignment)) % self.alignment
            current_offset += pad
            assigned_offset = current_offset

            tensor_infos.append({
                "hf_name": hf_name,
                "name": ggml_name,
                "n_dims": n_dims,
                "dims": dims_ggml,
                "type": ggml_type,
                "offset": assigned_offset,
                "pad": pad
            })
            current_offset += raw_len

        for info in tensor_infos:
            self._write_string(header_buf, info["name"])
            header_buf.write(struct.pack("<I", info["n_dims"]))
            for d in info["dims"]:
                header_buf.write(struct.pack("<Q", d))
            header_buf.write(struct.pack("<I", info["type"]))
            header_buf.write(struct.pack("<Q", info["offset"]))

        header_bytes = header_buf.getvalue()
        final_pad = (self.alignment - (len(header_bytes) % self.alignment)) % self.alignment

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"[*] Escribiendo encabezado GGUF ({len(header_bytes)} bytes) a disco...")
        with open(output_path, "wb") as out_f:
            out_f.write(header_bytes)
            if final_pad > 0:
                out_f.write(b"\x00" * final_pad)

            # Iterar archivo por archivo para minimizar aperturas
            for s_path in safetensor_paths:
                with safe_open(s_path, framework="pt", device="cpu") as sf:
                    file_keys = set(sf.keys())
                    matching_infos = [info for info in tensor_infos if info["hf_name"] in file_keys]
                    for info in matching_infos:
                        if info["pad"] > 0:
                            out_f.write(b"\x00" * info["pad"])

                        tensor = sf.get_tensor(info["hf_name"])
                        if info["type"] == GGML_TYPE_F32:
                            raw_bytes = tensor.to(torch.float32).numpy().tobytes()
                        elif info["type"] == GGML_TYPE_Q8_0:
                            raw_bytes = quantize_to_q8_0(tensor)
                        else:  # F16
                            raw_bytes = tensor.to(torch.float16).numpy().tobytes()

                        out_f.write(raw_bytes)
                        del tensor
                        del raw_bytes

                gc.collect()

        return {
            "output_path": output_path,
            "tensors_exported": len(tensor_infos),
            "quantization": quant_lower,
            "file_size_bytes": os.path.getsize(output_path),
        }
