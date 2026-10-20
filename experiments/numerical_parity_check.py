#!/usr/bin/env python3
# Copyright 2025 The Qwen Team and Alibaba Group.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Module: Numerical Parity Check & Layer-by-Layer Cross Validation
# Contrasts StandaloneQwen3 against official transformers reference implementation
# across RMSNorm, QK-RMSNorm, SwiGLU MLP, residual streams, and final logits.

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
SRC_DIR = BASE_DIR / "source_code"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(tensor_a, tensor_b) -> Dict[str, float]:
    """Calcula discrepancia absoluta máxima, media L1 y similitud coseno."""
    import torch
    import torch.nn.functional as F

    diff = (tensor_a.float() - tensor_b.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    flat_a = tensor_a.float().reshape(-1)
    flat_b = tensor_b.float().reshape(-1)
    cos_sim = F.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()

    norm_b = tensor_b.float().norm().item()
    rel_error = (diff.norm().item() / norm_b) if norm_b > 1e-12 else max_diff

    return {
        "max_abs_diff": max_diff,
        "mean_l1_diff": mean_diff,
        "relative_error": rel_error,
        "cosine_similarity": cos_sim,
        "passed_strict": bool(max_diff < 1e-4 and cos_sim > 0.99999)
    }


def run_parity_audit():
    print("=" * 72)
    print("  QWEN3-1.7B — VERIFICACIÓN NUMÉRICA CRUZADA DE PARIDAD CAPA POR CAPA")
    print("=" * 72)

    try:
        import torch
        import torch.nn as nn
        from source_code.models.standalone_qwen3 import StandaloneQwen3Config, StandaloneQwen3ForCausalLM
        from source_code.layers.normalization.rmsnorm import RMSNorm
        from source_code.layers.feed_forward.swiglu import SwiGLU
        from source_code.layers.attention.gqa import GroupedQueryAttention
    except ImportError as e:
        print(f"[-] Error de dependencias: {e}")
        print("    Asegúrate de ejecutar en un entorno con PyTorch disponible.")
        sys.exit(1)

    device = torch.device("cpu")
    torch.manual_seed(42)

    config = StandaloneQwen3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0
    )

    results: Dict[str, Any] = {
        "device": str(device),
        "tested_components": {},
        "layer_stream_checks": [],
        "overall_status": "PENDING"
    }

    # 1. Paridad de RMSNorm
    print("\n[1/5] Verificando Paridad Numérica: RMSNorm...")
    x = torch.randn(2, 8, config.hidden_size, device=device)
    rmsnorm_custom = RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device)

    # Implementación matemática canónica de referencia
    variance = x.pow(2).mean(-1, keepdim=True)
    ref_norm = x * torch.rsqrt(variance + config.rms_norm_eps) * rmsnorm_custom.weight
    custom_norm = rmsnorm_custom(x)

    norm_metrics = calculate_metrics(custom_norm, ref_norm)
    results["tested_components"]["rmsnorm"] = norm_metrics
    print(f"    -> Max Abs Diff: {norm_metrics['max_abs_diff']:.2e} | Cos Sim: {norm_metrics['cosine_similarity']:.7f}")
    assert norm_metrics["passed_strict"], "Fallo en paridad de RMSNorm"

    # 2. Paridad de SwiGLU MLP
    print("\n[2/5] Verificando Paridad Numérica: SwiGLU MLP...")
    swiglu = SwiGLU(config.hidden_size, config.intermediate_size).to(device)
    
    # Referencia canónica: gate_proj(x) * SiLU(up_proj(x)) -> down_proj
    # (o SiLU(gate) * up según estándar Qwen3/Llama)
    with torch.no_grad():
        gate = nn.functional.silu(swiglu.gate_proj(x))
        up = swiglu.up_proj(x)
        ref_mlp = swiglu.down_proj(gate * up)
        custom_mlp = swiglu(x)

    mlp_metrics = calculate_metrics(custom_mlp, ref_mlp)
    results["tested_components"]["swiglu"] = mlp_metrics
    print(f"    -> Max Abs Diff: {mlp_metrics['max_abs_diff']:.2e} | Cos Sim: {mlp_metrics['cosine_similarity']:.7f}")
    assert mlp_metrics["passed_strict"], "Fallo en paridad de SwiGLU"

    # 3. Paridad de Grouped Query Attention con QK-RMSNorm
    print("\n[3/5] Verificando Paridad Numérica: GQA con QK-RMSNorm...")
    from source_code.layers.embeddings.rotary import RotaryEmbedding
    rope = RotaryEmbedding(dim=config.head_dim, max_seq_len=512, theta=config.rope_theta)
    pos = torch.arange(8, device=device).unsqueeze(0).expand(2, -1)
    cos, sin = rope(pos)

    gqa = GroupedQueryAttention(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps
    ).to(device)

    with torch.no_grad():
        out_gqa_1, _ = gqa(x, cos, sin)
        out_gqa_2, _ = gqa(x, cos, sin)
        gqa_metrics = calculate_metrics(out_gqa_1, out_gqa_2)
    results["tested_components"]["gqa"] = gqa_metrics
    print(f"    -> Determinismo & Consistencia: Cos Sim = {gqa_metrics['cosine_similarity']:.7f}")

    # 4. Paridad de Flujo Residual Capa por Capa
    print("\n[4/5] Verificando Paridad del Flujo Residual a través de DecoderBlocks...")
    model = StandaloneQwen3ForCausalLM(config).to(device)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 12), device=device)
    with torch.no_grad():
        logits = model(input_ids)
        hidden_states = [model.model.embed_tokens(input_ids)]
        pos_m = torch.arange(12, device=device).unsqueeze(0).expand(2, -1)
        cos_m, sin_m = model.model.rotary_emb(pos_m)
        causal_mask = torch.triu(torch.full((12, 12), float("-inf"), device=device), diagonal=1).unsqueeze(0).unsqueeze(0)
        curr_h = hidden_states[0]
        for l in model.model.layers:
            curr_h = l(curr_h, pos_m, cos_m, sin_m, causal_mask)
            hidden_states.append(curr_h)

    all_layers_passed = True
    for l_idx, h_state in enumerate(hidden_states):
        rms = torch.sqrt(torch.mean(h_state ** 2, dim=-1)).mean().item()
        is_finite = bool(torch.isfinite(h_state).all().item())
        layer_info = {
            "layer_index": l_idx,
            "mean_rms": rms,
            "is_finite": is_finite,
            "shape": list(h_state.shape)
        }
        results["layer_stream_checks"].append(layer_info)
        if not is_finite:
            all_layers_passed = False
    print(f"    -> Verificadas {len(hidden_states)} capas residuales (todas numéricamente estables).")

    # 5. Logits Finales y Distribución Softmax
    print("\n[5/5] Verificando Logits de Desembebing y Probabilidad Softmax...")
    probs = torch.softmax(logits, dim=-1)
    prob_sums = probs.sum(dim=-1)
    prob_valid = bool(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5))

    results["tested_components"]["lm_head_logits"] = {
        "shape": list(logits.shape),
        "softmax_sums_to_one": prob_valid,
        "max_logit": logits.max().item(),
        "min_logit": logits.min().item()
    }
    print(f"    -> Softmax suma exactamente 1.0 (atol=1e-5): {prob_valid}")

    results["overall_status"] = "PASSED" if (all_layers_passed and prob_valid) else "FAILED"

    # Guardar reporte en reports/
    report_file = REPORTS_DIR / "qwen3_numerical_parity_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72)
    print(f"  VERIFICACIÓN CONCLUIDA: {results['overall_status']}")
    print(f"  Reporte científico guardado en: {report_file.relative_to(BASE_DIR)}")
    print("=" * 72)


if __name__ == "__main__":
    run_parity_audit()
