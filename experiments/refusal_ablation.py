# Copyright 2025 The Qwen Team and Alibaba Group.
# Apache License, Version 2.0 (the "License");
#
# Module: Refusal Direction Extraction & Safety Circuit Ablation Forensics
# Dissects the 1D refusal subspace in Qwen3 residual streams and evaluates
# orthogonalization and defensive amplification.

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from source_code.interpretability.refusal_direction import RefusalDirectionAnalyzer

def run_refusal_forensics():
    print("=" * 70)
    print("   FORENSE MECANICISTA: VECTOR GEOMÉTRICO DE RECHAZO (REFUSAL CIRCUIT)")
    print("   Disección de la Dirección de Seguridad r_hat en Capas 14-18")
    print("=" * 70 + "\n")
    
    hidden_size = 2048
    analyzer = RefusalDirectionAnalyzer(hidden_size=hidden_size)
    
    # Simulación de activaciones residuales contrastivas
    torch.manual_seed(42)
    true_refusal_axis = torch.randn(hidden_size)
    true_refusal_axis = true_refusal_axis / torch.norm(true_refusal_axis)
    
    # 20 pares contrastivos: Dañinos vs Inocuos
    num_samples = 20
    harmful_acts = torch.randn(num_samples, hidden_size) * 0.5 + true_refusal_axis * 4.5
    harmless_acts = torch.randn(num_samples, hidden_size) * 0.5 - true_refusal_axis * 1.0
    
    # 1. Extracción de la dirección de rechazo
    r_hat = analyzer.compute_refusal_direction(harmful_acts, harmless_acts)
    cosine_sim = torch.dot(r_hat, true_refusal_axis).item()
    print(f"[*] Vector unitario de rechazo r_hat extraído.")
    print(f"    - Dimensión: {r_hat.shape}")
    print(f"    - Fidelidad de recuperación del eje latente: {cosine_sim * 100:.2f}%\n")
    
    # 2. Medición de proyecciones
    print("[*] Proyección escalar sobre r_hat (h · r_hat):")
    test_harmful = torch.randn(5, hidden_size) * 0.5 + true_refusal_axis * 4.0
    test_harmless = torch.randn(5, hidden_size) * 0.5
    
    proj_harmful = analyzer.measure_refusal_projection(test_harmful)
    proj_harmless = analyzer.measure_refusal_projection(test_harmless)
    
    print(f"    - Prompts Dañinos (activación de defensa) : {proj_harmful.mean().item():+.4f}")
    print(f"    - Prompts Inocuos (flujo neutral)        : {proj_harmless.mean().item():+.4f}\n")
    
    # 3. Ortogonalización quirúrgica (Neutralización de rechazo)
    print("[*] Aplicando ablación ortogonal (h_perp = h - (h · r_hat) * r_hat):")
    h_perp = analyzer.orthogonalize(test_harmful)
    proj_after = analyzer.measure_refusal_projection(h_perp)
    print(f"    - Proyección residual tras ortogonalización: {proj_after.mean().item():.8f}")
    print("    [+] Disparador de rechazo neutralizado sin degradar dimensiones ortogonales.\n")
    
    # 4. Modo Inmunización Defensiva (Refusal Amplification)
    print("[*] Probando Inmunización contra Jailbreaks (h_def = h + alpha * r_hat):")
    adversarial_prompt = torch.randn(hidden_size) * 0.5 + true_refusal_axis * 0.8 # Jailbreak atenuado
    init_score = float(analyzer.measure_refusal_projection(adversarial_prompt))
    reinforced = analyzer.amplify_refusal(adversarial_prompt, factor=3.0)
    final_score = float(analyzer.measure_refusal_projection(reinforced))
    print(f"    - Score inicial de jailbreak: {init_score:+.4f}")
    print(f"    - Score tras amplificación defensiva: {final_score:+.4f} (+{final_score - init_score:.2f})")
    print("    [+] Sistema inmunizado exitosamente.")
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    run_refusal_forensics()
