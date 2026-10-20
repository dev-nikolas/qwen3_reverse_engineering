"""
Reverse Engineering Suite for Qwen3-1.7B
Análisis estructural, disección de pesos Safetensors, estadísticas tensoriales e inspección de activaciones internas.
"""

from __future__ import annotations
import json
import math
import struct
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def inspect_safetensors_binary_header(file_path: Path) -> Dict[str, Any]:
    """Realiza ingeniería inversa binaria sobre el encabezado del contenedor Safetensors."""
    with open(file_path, "rb") as f:
        # Los primeros 8 bytes corresponden a un unsigned 64-bit integer little-endian
        raw_header_size = f.read(8)
        header_size = struct.unpack("<Q", raw_header_size)[0]
        raw_header = f.read(header_size)
        header = json.loads(raw_header.decode("utf-8"))

    file_size = file_path.stat().st_size
    tensors_metadata = {k: v for k, v in header.items() if k != "__metadata__"}
    custom_metadata = header.get("__metadata__", {})

    dtypes = {}
    total_elements = 0
    for t_name, t_meta in tensors_metadata.items():
        dt = t_meta.get("dtype")
        dtypes[dt] = dtypes.get(dt, 0) + 1
        shape = t_meta.get("shape", [])
        numel = math.prod(shape) if shape else 0
        total_elements += numel

    return {
        "file_name": file_path.name,
        "file_size_bytes": file_size,
        "header_size_bytes": header_size,
        "payload_offset_bytes": 8 + header_size,
        "tensor_count": len(tensors_metadata),
        "dtypes_distribution": dtypes,
        "total_elements_in_shard": total_elements,
        "custom_metadata": custom_metadata,
        "sample_tensors": list(tensors_metadata.keys())[:10],
    }


def analyze_model_architecture_and_parameters() -> Dict[str, Any]:
    """Analiza la arquitectura formal de Qwen3-1.7B y desglosa los parámetros por subsistema."""
    config_path = MODEL_DIR / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    index_path = MODEL_DIR / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})

    # Contabilidad exacta de parámetros
    subsystem_params = {
        "embeddings": 0,
        "attention_qkv": 0,
        "attention_out": 0,
        "mlp_gate_up": 0,
        "mlp_down": 0,
        "layernorms": 0,
        "lm_head": 0,
        "other": 0,
    }

    tensor_details = []
    shards = set(weight_map.values())
    shard_readers = {s: safe_open(MODEL_DIR / s, framework="pt", device="cpu") for s in shards}

    total_params = 0
    unique_params = 0
    seen_storage_ptrs = set()

    for tensor_name, shard_file in weight_map.items():
        reader = shard_readers[shard_file]
        tensor_slice = reader.get_slice(tensor_name)
        shape = tensor_slice.get_shape()
        dtype = str(tensor_slice.get_dtype())
        numel = math.prod(shape)
        total_params += numel

        # Clasificación por subsistema
        if "embed_tokens" in tensor_name:
            subsystem_params["embeddings"] += numel
        elif any(x in tensor_name for x in ["q_proj", "k_proj", "v_proj"]):
            subsystem_params["attention_qkv"] += numel
        elif "o_proj" in tensor_name:
            subsystem_params["attention_out"] += numel
        elif any(x in tensor_name for x in ["gate_proj", "up_proj"]):
            subsystem_params["mlp_gate_up"] += numel
        elif "down_proj" in tensor_name:
            subsystem_params["mlp_down"] += numel
        elif "norm" in tensor_name:
            subsystem_params["layernorms"] += numel
        elif "lm_head" in tensor_name:
            subsystem_params["lm_head"] += numel
        else:
            subsystem_params["other"] += numel

        tensor_details.append({
            "name": tensor_name,
            "shape": shape,
            "numel": numel,
            "dtype": dtype,
            "shard": shard_file,
        })

    return {
        "config": config_data,
        "total_tensors": len(weight_map),
        "total_parameters_sum": total_params,
        "total_parameters_billions": round(total_params / 1e9, 4),
        "subsystem_parameters": subsystem_params,
        "subsystem_percentages": {
            k: f"{(v / total_params) * 100:.2f}%" for k, v in subsystem_params.items()
        },
        "sample_tensor_details": tensor_details[:15],
    }


def analyze_weights_numerical_distribution() -> Dict[str, Any]:
    """Calcula estadísticas numéricas reales (media, std, min, max, norma L2, dispersión) sobre capas críticas."""
    reader1 = safe_open(MODEL_DIR / "model-00001-of-00002.safetensors", framework="pt", device="cpu")
    reader2 = safe_open(MODEL_DIR / "model-00002-of-00002.safetensors", framework="pt", device="cpu")

    def get_tensor(name: str):
        if name in reader1.keys():
            return reader1.get_tensor(name).float()
        return reader2.get_tensor(name).float()

    inspected_layers = [0, 13, 27]  # Primera, intermedia y última capa
    layer_stats = {}

    for l_idx in inspected_layers:
        tensors_to_check = [
            f"model.layers.{l_idx}.input_layernorm.weight",
            f"model.layers.{l_idx}.self_attn.q_proj.weight",
            f"model.layers.{l_idx}.self_attn.k_proj.weight",
            f"model.layers.{l_idx}.self_attn.v_proj.weight",
            f"model.layers.{l_idx}.self_attn.o_proj.weight",
            f"model.layers.{l_idx}.post_attention_layernorm.weight",
            f"model.layers.{l_idx}.mlp.gate_proj.weight",
            f"model.layers.{l_idx}.mlp.up_proj.weight",
            f"model.layers.{l_idx}.mlp.down_proj.weight",
        ]
        
        l_stat = {}
        for t_name in tensors_to_check:
            t = get_tensor(t_name)
            short_name = t_name.split(f"layers.{l_idx}.")[-1]
            l_stat[short_name] = {
                "shape": list(t.shape),
                "mean": round(t.mean().item(), 6),
                "std": round(t.std().item(), 6),
                "min": round(t.min().item(), 6),
                "max": round(t.max().item(), 6),
                "l2_norm": round(t.norm().item(), 4),
                "sparsity_ratio": round((t == 0).float().mean().item(), 6),
            }
        layer_stats[f"layer_{l_idx}"] = l_stat

    # Embedding y Final Norm
    embed_t = get_tensor("model.embed_tokens.weight")
    final_norm = get_tensor("model.norm.weight")

    return {
        "embedding_stats": {
            "shape": list(embed_t.shape),
            "mean": round(embed_t.mean().item(), 6),
            "std": round(embed_t.std().item(), 6),
            "min": round(embed_t.min().item(), 6),
            "max": round(embed_t.max().item(), 6),
            "l2_norm": round(embed_t.norm().item(), 4),
        },
        "final_norm_stats": {
            "shape": list(final_norm.shape),
            "mean": round(final_norm.mean().item(), 6),
            "std": round(final_norm.std().item(), 6),
        },
        "layers": layer_stats,
    }


def inspect_internal_activations_and_inference() -> Dict[str, Any]:
    """Carga el modelo y tokenizer para capturar activaciones del residual stream mediante PyTorch forward hooks."""
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    
    # Cargar en torch.bfloat16 o float32 en CPU con baja memoria
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    prompt = "El núcleo cognitivo de Simple Domain audita datos deterministas"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Registro de hooks en el residual stream de capas seleccionadas
    captured_activations = {}

    def get_activation_hook(layer_name: str):
        def hook(module, input_args, output):
            # En DecoderLayer, output[0] es la salida del bloque residual (hidden_states)
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            with torch.no_grad():
                out_f = out.float()
                captured_activations[layer_name] = {
                    "shape": list(out_f.shape),
                    "mean_norm": round(out_f.norm(dim=-1).mean().item(), 4),
                    "std": round(out_f.std().item(), 4),
                    "max": round(out_f.max().item(), 4),
                    "min": round(out_f.min().item(), 4),
                }
        return hook

    hook_handles = []
    # Enganchar embeddings, capa 0, 7, 14, 21, 27
    hook_handles.append(model.model.embed_tokens.register_forward_hook(get_activation_hook("embedding")))
    for l_idx in [0, 7, 14, 21, 27]:
        hook_handles.append(
            model.model.layers[l_idx].register_forward_hook(get_activation_hook(f"layer_{l_idx}_residual"))
        )
    hook_handles.append(model.model.norm.register_forward_hook(get_activation_hook("final_layernorm")))

    # Ejecutar inferencia de 1 token para recolectar las activaciones
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    latency_ms = round((time.time() - t0) * 1000, 2)

    logits = outputs.logits
    next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
    next_token_str = tokenizer.decode([next_token_id])

    # Remover hooks
    for h in hook_handles:
        h.remove()

    return {
        "test_prompt": prompt,
        "token_count": inputs.input_ids.shape[1],
        "forward_latency_ms": latency_ms,
        "predicted_next_token_id": next_token_id,
        "predicted_next_token_text": next_token_str,
        "logits_shape": list(logits.shape),
        "captured_residual_stream": captured_activations,
    }


def main():
    print("=== [1/4] Inspeccionando encabezados binarios Safetensors ===")
    shard1_meta = inspect_safetensors_binary_header(MODEL_DIR / "model-00001-of-00002.safetensors")
    shard2_meta = inspect_safetensors_binary_header(MODEL_DIR / "model-00002-of-00002.safetensors")

    print("=== [2/4] Analizando arquitectura y desglose de parámetros ===")
    arch_summary = analyze_model_architecture_and_parameters()

    print("=== [3/4] Disección y estadísticas numéricas de pesos ===")
    weight_stats = analyze_weights_numerical_distribution()

    print("=== [4/4] Inferencia e inspección de activaciones del residual stream ===")
    activation_results = inspect_internal_activations_and_inference()

    full_report = {
        "model_id": "Qwen/Qwen3-1.7B",
        "license": "Apache-2.0",
        "binary_shards": [shard1_meta, shard2_meta],
        "architecture_summary": arch_summary,
        "weights_distribution": weight_stats,
        "activation_probing": activation_results,
    }

    report_path = OUTPUT_DIR / "qwen3_reverse_engineering_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"Reporte generado exitosamente en: {report_path}")


if __name__ == "__main__":
    main()
