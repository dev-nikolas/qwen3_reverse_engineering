import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
os.makedirs(REPORTS_DIR, exist_ok=True)

COMPARISON_DATA = {
    "architectures": {
        "Qwen3-1.7B": {
            "creator": "Alibaba Cloud",
            "parameters": "1.72B",
            "layers": 28,
            "hidden_size": 2048,
            "attention_mechanism": "GQA (16 Query, 8 KV heads)",
            "head_dim": 128,
            "qk_norm": True,
            "mlp_type": "SwiGLU",
            "intermediate_size": 6144,
            "rope_theta": 1000000.0,
            "vocab_size": 151936,
            "native_reasoning_tokens": True,
            "logit_softcapping": False,
            "advantages": [
                "QK-Norm estabiliza el gradiente y previene explosión de atención en contextos largos.",
                "RoPE theta 1,000,000 permite escalar hasta 131k tokens con mínima degradación.",
                "Tokens nativos de pensamiento (<think>) integrados a nivel de pre-entrenamiento.",
                "Excelente balance de compresión GQA 2:1 que preserva resolución frente a 4:1."
            ]
        },
        "Llama-3.2-1B": {
            "creator": "Meta AI",
            "parameters": "1.23B",
            "layers": 16,
            "hidden_size": 2048,
            "attention_mechanism": "GQA (32 Query, 8 KV heads)",
            "head_dim": 64,
            "qk_norm": False,
            "mlp_type": "SwiGLU",
            "intermediate_size": 8192,
            "rope_theta": 500000.0,
            "vocab_size": 128256,
            "native_reasoning_tokens": False,
            "logit_softcapping": False,
            "advantages": [
                "Capas más reducidas (16), menor latencia por token.",
                "Head dim pequeño (64) que reduce el tamaño del KV cache."
            ]
        },
        "Gemma-2-2B": {
            "creator": "Google DeepMind",
            "parameters": "2.61B",
            "layers": 26,
            "hidden_size": 2304,
            "attention_mechanism": "GQA Alternada (Local 4k / Global)",
            "head_dim": 256,
            "qk_norm": False,
            "mlp_type": "GeGLU",
            "intermediate_size": 9216,
            "rope_theta": 10000.0,
            "vocab_size": 256000,
            "native_reasoning_tokens": False,
            "logit_softcapping": True,
            "advantages": [
                "Logit soft-capping previene que los logits se desborden.",
                "Arquitectura dual pre/post-norm y atención por ventana deslizante."
            ]
        },
        "DeepSeek-V3/R1 (MoE)": {
            "creator": "DeepSeek AI",
            "parameters": "671B (37B activos)",
            "layers": 61,
            "hidden_size": 2048,
            "attention_mechanism": "MLA (Multi-Head Latent Attention)",
            "head_dim": 128,
            "qk_norm": True,
            "mlp_type": "SwiGLU MoE (256 expertos)",
            "intermediate_size": 2048,
            "rope_theta": 10000.0,
            "vocab_size": 129280,
            "native_reasoning_tokens": True,
            "logit_softcapping": False,
            "advantages": [
                "MLA comprime KV a un vector latente de baja dimensión (ahorro masivo de RAM).",
                "Especialización por mezcla de expertos (MoE) ultra-eficiente."
            ]
        }
    },
    "qwen3_layer_band_forensics": {
        "band_1_syntactic_foundation": {
            "layers": "00 a 04",
            "role": "Substrato Sintáctico y Normalización de Entrada",
            "mechanism": "Calcula el embedding posicional angular de alta frecuencia (RoPE) y resuelve relaciones sintácticas primitivas (preposiciones, artículos, concordancia de género).",
            "ablation_drop": "94.1% a 97.6% (Colapso total del modelo)",
            "verdict": "Órganos vitales no podables."
        },
        "band_2_semantic_fusion": {
            "layers": "05 a 11",
            "role": "Fusión Semántica y Desambiguación Léxica",
            "mechanism": "Las cabezas de atención vinculan sustantivos con sus atributos adjetivales y relaciones sujeto-verbo.",
            "ablation_drop": "17.0% a 67.5%",
            "verdict": "Capas críticas para comprensión contextual."
        },
        "band_3_memory_and_induction": {
            "layers": "12 a 18",
            "role": "Almacenamiento de Conocimiento Asociativo y Circuitos de Inducción",
            "mechanism": "Aquí operan los diccionarios de memoria en los MLPs (ej. neurona 1950 en L18) y el circuito copiador de inducción principal (L16H14 con traza +15.82). Es el centro neurálgico donde el trasplante causal desvía respuestas.",
            "ablation_drop": "Sensibilidad moderada a alta (26.1% a 52.2%)",
            "verdict": "Epicentro del conocimiento fáctico y razonamiento en contexto."
        },
        "band_4_reasoning_and_logit_lens": {
            "layers": "19 a 24",
            "role": "Consolidación de Respuesta y Ensambles Redundantes",
            "mechanism": "Circuitos de respaldo de inducción (L20H14, L21H8, L21H9). En el Logit Lens, la predicción pasa de 0% (L16) a 44.7% (L21) y 99.6% (L24). La norma residual alcanza su pico de energía máxima (1534.82).",
            "ablation_drop": "0.8% a 41.4%",
            "verdict": "Punto de colapso de incertidumbre donde el modelo decide el token final."
        },
        "band_5_lexical_refinement": {
            "layers": "25 a 27",
            "role": "Refinamiento Léxico, Inhibición y Proyección al Vocabulario",
            "mechanism": "La Capa 26 es 100% redundante (+1.1% si se retira). La Capa 27 contrae la norma de 1534.82 a 99.21 para estabilizar el paso por RMSNorm y entrar limpiamente a la matriz lm_head de 151,936 tokens.",
            "ablation_drop": "-1.1% a 2.1%",
            "verdict": "Capas superficiales altamente podables mediante Layer Pruning."
        }
    }
}

def run_comparison():
    print("=" * 70)
    print("  ANÁLISIS COMPARATIVO DE ARQUITECTURAS Y BANDA DE CAPAS")
    print("=" * 70)
    
    print("\n--- 1. CUADRO COMPARATIVO: QWEN3-1.7B VS COMPETIDORES DIRECTOS ---")
    header = f"{'Modelo':<16} | {'Params':<8} | {'Capas':<6} | {'GQA / Attn':<22} | {'QK-Norm':<8} | {'RoPE Theta':<10}"
    print(header)
    print("-" * 78)
    for model_name, info in COMPARISON_DATA["architectures"].items():
        qk_str = "SÍ" if info["qk_norm"] else "NO"
        print(f"{model_name:<16} | {info['parameters']:<8} | {info['layers']:<6} | {info['attention_mechanism']:<22} | {qk_str:<8} | {int(info['rope_theta']):<10}")

    print("\n--- 2. RADIOGRAFÍA FUNCIONAL DE LAS 28 CAPAS DE QWEN3 ---")
    for band_key, band in COMPARISON_DATA["qwen3_layer_band_forensics"].items():
        print(f"\n[Banda: {band['layers']}] -> {band['role']}")
        print(f"  * Mecánica Interna : {band['mechanism']}")
        print(f"  * Impacto Ablación : {band['ablation_drop']}")
        print(f"  * Veredicto        : {band['verdict']}")

    # Guardamos el informe en JSON
    output_file = os.path.join(REPORTS_DIR, "architectural_comparison.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(COMPARISON_DATA, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 70)
    print(f"[+] Informe comparativo guardado en: reports/architectural_comparison.json")
    print("=" * 70)

if __name__ == "__main__":
    run_comparison()
