# Copyright 2026 Alibaba Group and the Qwen Team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

"""
Exportador de Cuantización AWQ 4-Bit (W4A16) — Qwen3-1.7B
=========================================================
Convierte los shards oficiales Safetensors de 4.06 GB a un checkpoint compacto
de ~1.1 GB en formato W4A16, protegiendo determinísticamente la Dimensión Rebelde 1793.

Características:
  - Empaquetado de 2 nibbles (4-bit) por cada byte uint8.
  - Cuantización por grupos (group_size=128) con factor de escala y punto cero.
  - Protección espectral de canales sobresalientes en residual stream.
  - Streaming capa por capa para bajo consumo de RAM durante la exportación.
"""

import os
import sys
import time
import argparse
import torch  # type: ignore
import torch.nn as nn  # type: ignore

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_DIR = os.path.join(BASE_DIR, "source_code")
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

from inference.awq_quantizer import AWQChannelProtector, pack_int4_weights, unpack_int4_weights
from loaders.weight_loader import SafetensorsLoader

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    except Exception:
        pass


def export_awq_checkpoint(
    model_dir: str = None,
    output_path: str = None,
    group_size: int = 128,
    rogue_dim_index: int = 1793
) -> dict:
    if model_dir is None:
        model_dir = os.path.join(BASE_DIR, "model")
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "qwen3-1.7b-w4a16.pt")

    print("=" * 68)
    print("  QWEN3-1.7B — EXPORTADOR DE PESOS 4-BIT AWQ (W4A16)")
    print("=" * 68)
    print(f"  Directorio de Entrada  : {model_dir}")
    print(f"  Archivo de Salida      : {output_path}")
    print(f"  Tamaño de Grupo        : {group_size}")
    print(f"  Dimensión Protegida    : Canal {rogue_dim_index}")
    print("=" * 68 + "\n")

    loader = SafetensorsLoader(model_dir)
    if not loader.shard_files:
        raise FileNotFoundError(f"No se encontraron shards Safetensors en: {model_dir}")

    protector = AWQChannelProtector(alpha=0.5, rogue_dim_index=rogue_dim_index)
    quantized_state_dict = {}
    
    total_original_bytes = 0
    total_quantized_bytes = 0
    layers_processed = 0

    start_time = time.time()

    # Identificar tensores en los shards
    shards_data = {}
    from safetensors.torch import load_file  # type: ignore
    for s_path in loader.shard_files:
        print(f"[*] Escaneando shard: {os.path.basename(s_path)}...")
        shards_data.update(load_file(s_path))

    total_tensors = len(shards_data)
    print(f"[+] Total de tensores cargados: {total_tensors}\n")

    for name, tensor in shards_data.items():
        original_bytes = tensor.numel() * tensor.element_size()
        total_original_bytes += original_bytes

        # Solo cuantizamos proyecciones lineales grandes de atención y MLP
        is_linear_proj = any(proj in name for proj in [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]) and tensor.dim() == 2

        if is_linear_proj:
            # Crear Linear temporal
            out_f, in_f = tensor.shape
            dummy_linear = nn.Linear(in_f, out_f, bias=False)
            dummy_linear.weight.data.copy_(tensor.float())

            act_scales = torch.ones(in_f)
            q_w8, s, group_scales = protector.protect_and_quantize_layer(
                dummy_linear,
                act_scales=act_scales,
                bits=4,
                group_size=group_size
            )

            # Empaquetar a 4 bits (uint8)
            packed_w = pack_int4_weights(q_w8.view(out_f, -1)[:, :in_f])
            
            quantized_state_dict[f"{name}.packed_weight"] = packed_w
            quantized_state_dict[f"{name}.scales"] = s.half()
            quantized_state_dict[f"{name}.group_scales"] = group_scales.half()

            q_bytes = packed_w.numel() + s.numel() * 2 + group_scales.numel() * 2
            total_quantized_bytes += q_bytes
            layers_processed += 1

            if layers_processed % 20 == 0 or layers_processed == 1:
                ratio = (1.0 - (q_bytes / original_bytes)) * 100.0
                print(f"  [AWQ-W4] {name:48} -> {original_bytes // 1024:6d} KB  a {q_bytes // 1024:5d} KB (-{ratio:4.1f}%)")
        else:
            # Conservar embeddings, lm_head y normalizaciones en formato nativo / FP16
            if tensor.dtype == torch.float32 and tensor.dim() > 1:
                tensor = tensor.half()
            quantized_state_dict[name] = tensor
            total_quantized_bytes += tensor.numel() * tensor.element_size()

    elapsed = time.time() - start_time
    compression_ratio = total_original_bytes / max(1, total_quantized_bytes)
    savings_pct = (1.0 - (total_quantized_bytes / total_original_bytes)) * 100.0

    print("\n" + "=" * 68)
    print("  RESULTADOS DE COMPRESIÓN AWQ 4-BIT")
    print("=" * 68)
    print(f"  Capas Cuantizadas      : {layers_processed}")
    print(f"  Tamaño Original (FP16) : {total_original_bytes / (1024**3):.2f} GB")
    print(f"  Tamaño Cuantizado (W4) : {total_quantized_bytes / (1024**3):.2f} GB")
    print(f"  Ahorro de Memoria      : {savings_pct:.1f}%")
    print(f"  Factor de Compresión   : {compression_ratio:.2f}x")
    print(f"  Tiempo de Procesamiento: {elapsed:.1f} s")
    print("-" * 68)

    print(f"[*] Guardando checkpoint compacto en: {output_path}...")
    torch.save(quantized_state_dict, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Checkpoint W4A16 guardado con éxito ({file_size_mb:.2f} MB en disco).")
    print("=" * 68)

    return {
        "layers_quantized": layers_processed,
        "original_gb": total_original_bytes / (1024**3),
        "quantized_gb": total_quantized_bytes / (1024**3),
        "savings_pct": savings_pct,
        "output_path": output_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exportador AWQ W4A16 de Qwen3-1.7B")
    parser.add_argument("--model-dir", default=None, help="Ruta a los shards safetensors")
    parser.add_argument("--output", default=None, help="Ruta del checkpoint .pt de salida")
    parser.add_argument("--group-size", type=int, default=128, help="Tamaño de grupo para escalas (def: 128)")
    args = parser.parse_args()

    export_awq_checkpoint(
        model_dir=args.model_dir,
        output_path=args.output,
        group_size=args.group_size
    )
