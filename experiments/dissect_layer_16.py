"""
Deep Dissection of Layer 16 and Induction Head 14 in Qwen3-1.7B
1. Descomposición del Circuito QK y Circuito OV (Query-Key / Output-Value Circuit Analysis).
2. Espectro y Traza de W_OV: ¿Actúa como operador de copia de identidad pura?
3. Experimento de Ablación Causal (Knockout Test): Comportamiento del modelo con el Cabezal 14 desactivado (pérdida de capacidad de copia en contexto).
"""

from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_qk_and_ov_circuits(model) -> Dict[str, Any]:
    """
    Descompone matemáticamente los sub-circuitos QK y OV del Cabezal 14 (Capa 16).
    En la teoría de 'Transformer Circuits' (Anthropic):
    - W_QK = W_Q^h @ (W_K^h)^T determina DÓNDE atiende el cabezal.
    - W_OV = W_V^h @ W_O^h determina QUÉ INFORMACIÓN se copia al residual stream.
    """
    layer = model.model.layers[16]
    head_dim = 128
    head_idx = 14
    # GQA: 16 cabezas de Query, 8 cabezas de Key/Value.
    # El mapeo de cabeza KV para la cabeza Q 14 es kv_head_idx = head_idx // (num_heads // num_kv_heads) = 14 // 2 = 7.
    kv_head_idx = head_idx // 2

    # Extraer pesos de las proyecciones
    # W_q: [2048, 2048]
    # W_k: [1024, 2048]
    # W_v: [1024, 2048]
    # W_o: [2048, 2048]
    W_q = layer.self_attn.q_proj.weight.data.float()
    W_k = layer.self_attn.k_proj.weight.data.float()
    W_v = layer.self_attn.v_proj.weight.data.float()
    W_o = layer.self_attn.o_proj.weight.data.float()

    # Rebanar la cabeza específica
    q_start = head_idx * head_dim
    q_end = (head_idx + 1) * head_dim
    W_q_head = W_q[q_start:q_end, :]  # [128, 2048]

    kv_start = kv_head_idx * head_dim
    kv_end = (kv_head_idx + 1) * head_dim
    W_k_head = W_k[kv_start:kv_end, :]  # [128, 2048]
    W_v_head = W_v[kv_start:kv_end, :]  # [128, 2048]

    W_o_head = W_o[:, q_start:q_end]    # [2048, 128]

    # 1. Circuito OV: W_OV = W_v_head.T @ W_o_head.T en espacio de residual stream [2048, 2048]
    # Operación de transmisión: x -> x @ W_v_head.T @ W_o_head.T
    # Matriz efectiva W_OV: W_o_head @ W_v_head -> shape [2048, 2048]
    W_ov = W_o_head @ W_v_head  # [2048, 2048]
    
    # 2. Circuito QK: W_qk = W_q_head.T @ W_k_head -> shape [2048, 2048]
    W_qk = W_q_head.T @ W_k_head  # [2048, 2048]

    # Calcular valores singulares y autovalores de W_OV
    svd_ov = torch.linalg.svdvals(W_ov)
    trace_ov = torch.trace(W_ov).item()
    
    # Autovalores para comprobar si es un operador de copia positivo (eigenvalues con parte real > 0)
    eigenvals_ov = torch.linalg.eigvals(W_ov)
    real_parts = eigenvals_ov.real
    positive_eigenvals_ratio = (real_parts > 0).float().mean().item()

    # Rango efectivo de W_OV (máximo 128 porque el cuello de botella es head_dim=128)
    s_norm = svd_ov[:head_dim] / svd_ov[:head_dim].sum()
    eff_rank_ov = math.exp(-torch.sum(s_norm * torch.log(s_norm + 1e-12)).item())

    return {
        "head_index": head_idx,
        "kv_head_mapped_index": kv_head_idx,
        "head_dimension": head_dim,
        "trace_w_ov": round(trace_ov, 4),
        "positive_eigenvalues_ratio": round(positive_eigenvals_ratio, 4),
        "max_singular_val_ov": round(svd_ov[0].item(), 4),
        "effective_rank_ov": round(eff_rank_ov, 2),
        "max_rank_ov": head_dim,
        "is_copying_circuit": bool(positive_eigenvals_ratio > 0.5),
        "first_5_singular_vals_ov": [round(x.item(), 4) for x in svd_ov[:5]],
        "first_5_singular_vals_qk": [round(x.item(), 4) for x in torch.linalg.svdvals(W_qk)[:5]],
    }


def run_ablation_knockout_experiment(model, tokenizer) -> Dict[str, Any]:
    """
    Experimento de Ablación Causal (Knockout Test):
    Compara la capacidad del modelo para predecir la repetición de una secuencia
    en dos condiciones:
    Condición 1: Modelo normal (baseline intacto).
    Condición 2: Cabezal 14 de la Capa 16 mutilado/desactivado (ablated con hook cero).
    """
    test_sequence = "Las constantes cosmológicas son: Lambda, Gamma, Omega, Phi, Kappa. De nuevo: Lambda, Gamma, Omega, Phi,"
    expected_continuation = " Kappa"

    inputs = tokenizer(test_sequence, return_tensors="pt")
    target_token_id = tokenizer.encode(expected_continuation, add_special_tokens=False)[0]

    # --- 1. Ejecución Base (Intacto) ---
    with torch.no_grad():
        out_intact = model(**inputs)
    logits_intact = out_intact.logits[0, -1, :].float()
    probs_intact = F.softmax(logits_intact, dim=-1)
    prob_target_intact = probs_intact[target_token_id].item()
    top_1_token_intact = tokenizer.decode([torch.argmax(logits_intact).item()])

    # --- 2. Ejecución con Ablación de la Capa 16, Cabezal 14 ---
    # Interceptamos la salida de atención antes de o_proj o directamente en o_proj
    head_dim = 128
    head_idx = 14
    q_start = head_idx * head_dim
    q_end = (head_idx + 1) * head_dim

    def ablation_hook(module, inp, out):
        # out es la salida de self_attn (hidden_states o tuple)
        # Para anular exactamente el cabezal 14, ponemos a cero sus activaciones intermedias
        if isinstance(out, tuple):
            attn_out = out[0].clone()
        else:
            attn_out = out.clone()
        # Modificar solo la componente del cabezal 14
        # En la salida final de la capa de atención, o_proj ya proyectó.
        # Por tanto, podemos hookear o_proj directamente
        return out

    # Hook en o_proj: entrada a o_proj es [batch, seq_len, 2048]
    def o_proj_ablation_pre_hook(module, args):
        x = args[0].clone()
        # x tiene shape [1, seq_len, 2048]
        # Ponemos a cero los canales correspondientes al cabezal 14
        x[:, :, q_start:q_end] = 0.0
        return (x,)

    hook_handle = model.model.layers[16].self_attn.o_proj.register_forward_pre_hook(o_proj_ablation_pre_hook)

    with torch.no_grad():
        out_ablated = model(**inputs)
    logits_ablated = out_ablated.logits[0, -1, :].float()
    probs_ablated = F.softmax(logits_ablated, dim=-1)
    prob_target_ablated = probs_ablated[target_token_id].item()
    top_1_token_ablated = tokenizer.decode([torch.argmax(logits_ablated).item()])

    hook_handle.remove()

    prob_drop_pct = ((prob_target_intact - prob_target_ablated) / (prob_target_intact + 1e-12)) * 100

    return {
        "prompt_evaluated": test_sequence,
        "expected_token": expected_continuation,
        "intact_model": {
            "top_1_predicted": top_1_token_intact,
            "target_token_prob": round(prob_target_intact, 4),
            "target_token_prob_pct": f"{prob_target_intact * 100:.2f}%",
        },
        "ablated_model_head_14_zeroed": {
            "top_1_predicted": top_1_token_ablated,
            "target_token_prob": round(prob_target_ablated, 4),
            "target_token_prob_pct": f"{prob_target_ablated * 100:.2f}%",
        },
        "causal_impact": {
            "absolute_prob_drop": round(prob_target_intact - prob_target_ablated, 4),
            "relative_prob_drop_pct": f"{prob_drop_pct:.2f}%",
            "is_causally_critical": bool(prob_drop_pct > 15.0),
        },
    }


def main():
    print("=== [1] Cargando modelo Qwen3-1.7B en memoria para disección de Capa 16 ===")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Modelo cargado en {time.time() - t0:.2f} s")

    print("\n=== [2] Análisis Matemático de Circuitos QK y OV (Capa 16, Cabezal 14) ===")
    circuit_data = analyze_qk_and_ov_circuits(model)
    print(f"  Traza de W_OV: {circuit_data['trace_w_ov']}")
    print(f"  Proporción de autovalores reales positivos: {circuit_data['positive_eigenvalues_ratio']*100:.1f}%")
    print(f"  Rango efectivo de W_OV: {circuit_data['effective_rank_ov']} / {circuit_data['max_rank_ov']}")
    print(f"  ¿Es un circuito de copia de identidad?: {'SÍ' if circuit_data['is_copying_circuit'] else 'NO'}")

    print("\n=== [3] Experimento de Ablación Causal (Knockout Test) ===")
    ablation_results = run_ablation_knockout_experiment(model, tokenizer)
    print(f"  Probabilidad con modelo INTACTO: {ablation_results['intact_model']['target_token_prob_pct']}")
    print(f"  Probabilidad con Cabezal 14 MUTILADO (cero): {ablation_results['ablated_model_head_14_zeroed']['target_token_prob_pct']}")
    print(f"  Caída relativa de probabilidad: {ablation_results['causal_impact']['relative_prob_drop_pct']}")
    print(f"  ¿Es causalmente crítico?: {'SÍ' if ablation_results['causal_impact']['is_causally_critical'] else 'NO'}")

    dissection_report = {
        "target_component": "Layer_16_Head_14",
        "ov_and_qk_circuit_analysis": circuit_data,
        "causal_ablation_experiment": ablation_results,
    }

    out_file = REPORT_DIR / "qwen3_layer16_head14_deep_dissection.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dissection_report, f, indent=2, ensure_ascii=False)

    print(f"\nReporte completo de disección de la Capa 16 guardado en: {out_file}")


if __name__ == "__main__":
    main()
