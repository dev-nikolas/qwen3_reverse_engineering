"""
Advanced Reverse Engineering Suite for Qwen3-1.7B
1. Logit Lens: Decodificación capa por capa de la formación del pensamiento.
2. Singular Value Decomposition (SVD): Espectro de valores singulares y rango efectivo de matrices de pesos.
3. Chat Template y Thinking Mode Inspection.
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


def run_logit_lens_investigation(model, tokenizer, prompt: str) -> List[Dict[str, Any]]:
    """Logit Lens: Proyecta el residual stream de CADA capa (0..27) a través de lm_head para ver cómo madura la predicción."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    hidden_states_per_layer = []
    
    def hook_fn(layer_idx):
        def _hook(module, input_args, output):
            h = output[0] if isinstance(output, tuple) else output
            hidden_states_per_layer.append((layer_idx, h.detach().clone()))
        return _hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(hook_fn(i)))

    with torch.no_grad():
        outputs = model(**inputs)

    for h in handles:
        h.remove()

    last_token_idx = inputs.input_ids.shape[1] - 1
    lens_results = []

    # Proyección a través de model.norm y lm_head
    for layer_idx, h in hidden_states_per_layer:
        h_last = h[:, last_token_idx:last_token_idx+1, :].float()
        normed = model.model.norm(h_last.to(model.model.norm.weight.dtype))
        logits = model.lm_head(normed).float().squeeze()
        probs = F.softmax(logits, dim=-1)
        top_k = torch.topk(probs, k=5)
        
        top_tokens = []
        for val, idx in zip(top_k.values, top_k.indices):
            top_tokens.append({
                "token": tokenizer.decode([idx.item()]),
                "token_id": idx.item(),
                "probability": round(val.item(), 4),
            })
        
        lens_results.append({
            "layer": layer_idx,
            "top_1_token": top_tokens[0]["token"],
            "top_1_prob": top_tokens[0]["probability"],
            "top_5_candidates": top_tokens,
        })

    return lens_results


def run_svd_weight_spectrum_analysis(model) -> Dict[str, Any]:
    """Calcula el decaimiento de valores singulares (SVD) y el rango efectivo en capas representativas."""
    inspected_tensors = {
        "layer_0_q_proj": model.model.layers[0].self_attn.q_proj.weight.data.float(),
        "layer_0_gate_proj": model.model.layers[0].mlp.gate_proj.weight.data.float(),
        "layer_14_q_proj": model.model.layers[14].self_attn.q_proj.weight.data.float(),
        "layer_14_gate_proj": model.model.layers[14].mlp.gate_proj.weight.data.float(),
        "layer_27_q_proj": model.model.layers[27].self_attn.q_proj.weight.data.float(),
        "layer_27_down_proj": model.model.layers[27].mlp.down_proj.weight.data.float(),
    }

    svd_summary = {}

    for name, tensor in inspected_tensors.items():
        # U, S, V = torch.linalg.svd(tensor, full_matrices=False)
        S = torch.linalg.svdvals(tensor)
        s_norm = S / S.sum()
        # Entropía espectral y rango efectivo (Roy & Vetterli)
        spectral_entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-12)).item()
        effective_rank = math.exp(spectral_entropy)
        
        # Porcentaje de energía en los primeros 10% de valores singulares
        k_10 = max(1, int(len(S) * 0.1))
        energy_top_10_pct = (torch.sum(S[:k_10] ** 2) / torch.sum(S ** 2)).item()

        svd_summary[name] = {
            "matrix_shape": list(tensor.shape),
            "max_singular_val": round(S[0].item(), 4),
            "min_singular_val": round(S[-1].item(), 4),
            "condition_number": round((S[0] / (S[-1] + 1e-12)).item(), 2),
            "effective_rank": round(effective_rank, 2),
            "max_possible_rank": min(tensor.shape),
            "rank_utilization_pct": f"{(effective_rank / min(tensor.shape)) * 100:.2f}%",
            "energy_in_top_10pct_sv": round(energy_top_10_pct * 100, 2),
            "first_5_singular_values": [round(x.item(), 4) for x in S[:5]],
        }

    return svd_summary


def inspect_chat_template_and_special_tokens(tokenizer) -> Dict[str, Any]:
    """Disecciona el protocolo de instrucciones, chat template y tokens de pensamiento/sistema."""
    special_tokens = tokenizer.special_tokens_map
    vocab = tokenizer.get_vocab()
    
    # Identificar tokens especiales de razonamiento/sistema
    thinking_tokens = {}
    for tok_candidate in ["<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        if tok_candidate in vocab:
            thinking_tokens[tok_candidate] = vocab[tok_candidate]

    # Renderizar un mensaje de prueba con el chat template
    messages = [
        {"role": "system", "content": "Eres un núcleo de análisis determinista."},
        {"role": "user", "content": "Explica la verificación de integridad de datos."},
    ]
    has_chat_template = hasattr(tokenizer, "chat_template") and bool(tokenizer.chat_template)
    rendered_chat = tokenizer.apply_chat_template(messages, tokenize=False) if has_chat_template else None

    return {
        "vocab_size": len(vocab),
        "special_tokens": special_tokens,
        "identified_control_tokens": thinking_tokens,
        "has_chat_template": has_chat_template,
        "rendered_sample_prompt": rendered_chat,
    }


def main():
    print("=== [A] Cargando modelo en memoria para análisis avanzado ===")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Modelo cargado en {time.time() - t0:.2f} s")

    print("=== [B] Ejecutando Logit Lens (Inspección capa por capa) ===")
    prompt = "El lenguaje de programación más utilizado en la ciencia de datos es"
    logit_lens_data = run_logit_lens_investigation(model, tokenizer, prompt)
    
    print("\n--- Resultados Logit Lens (Predicción del próximo token por capa) ---")
    for step in logit_lens_data[::3]:  # Mostrar cada 3 capas
        print(f"Capa {step['layer']:2d}: Predice {json.dumps(step['top_1_token'])} con prob {step['top_1_prob']*100:.1f}%")
    print(f"Capa 27: Predice {json.dumps(logit_lens_data[-1]['top_1_token'])} con prob {logit_lens_data[-1]['top_1_prob']*100:.1f}%\n")


    print("=== [C] Descomposición en Valores Singulares (SVD) de Pesos ===")
    svd_data = run_svd_weight_spectrum_analysis(model)

    print("=== [D] Inspección de Protocolo de Tokens y Chat Template ===")
    token_proto = inspect_chat_template_and_special_tokens(tokenizer)

    advanced_report = {
        "logit_lens_experiment": {
            "prompt": prompt,
            "layers_evolution": logit_lens_data,
        },
        "svd_spectral_analysis": svd_data,
        "protocol_and_tokenizer": token_proto,
    }

    out_path = REPORT_DIR / "qwen3_advanced_reverse_engineering.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(advanced_report, f, indent=2, ensure_ascii=False)

    print(f"\nReporte avanzado guardado en: {out_path}")


if __name__ == "__main__":
    main()
