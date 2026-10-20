import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COMMANDS = {
    "chat": os.path.join(BASE_DIR, "tools", "chat.py"),
    "standalone-chat": os.path.join(BASE_DIR, "tools", "standalone_chat.py"),
    "benchmark": os.path.join(BASE_DIR, "tools", "benchmark_engine.py"),
    "inspect": os.path.join(BASE_DIR, "tools", "interactive_inspector.py"),
    "test": os.path.join(BASE_DIR, "source_code", "tests", "test_architecture.py"),
    "refusal": os.path.join(BASE_DIR, "experiments", "refusal_ablation.py"),
    "surgery": os.path.join(BASE_DIR, "experiments", "knowledge_surgery.py"),
    "causal": os.path.join(BASE_DIR, "experiments", "causal_deep_forensics.py"),
    "induction": os.path.join(BASE_DIR, "experiments", "deep_circuit_forensics.py"),
    "dissect": os.path.join(BASE_DIR, "experiments", "dissect_layer_16.py"),
    "logit-lens": os.path.join(BASE_DIR, "experiments", "advanced_reverse_engineering.py"),
    "binary": os.path.join(BASE_DIR, "experiments", "reverse_engineer_qwen3.py"),
    "compare": os.path.join(BASE_DIR, "experiments", "architectural_comparison.py"),
    "steer": os.path.join(BASE_DIR, "experiments", "activation_steering.py"),
    "tokenizer": os.path.join(BASE_DIR, "experiments", "tokenizer_forensics.py"),
    "heads": os.path.join(BASE_DIR, "experiments", "classify_448_heads.py"),
    "spectral": os.path.join(BASE_DIR, "experiments", "spectral_dimension.py"),
    "suppression": os.path.join(BASE_DIR, "experiments", "copy_suppression_forensics.py"),
    "export-gguf": os.path.join(BASE_DIR, "tools", "export_gguf.py"),
    "parity": os.path.join(BASE_DIR, "experiments", "numerical_parity_check.py"),
    "download": os.path.join(BASE_DIR, "tools", "download_weights.py"),
    "induction-bench": os.path.join(BASE_DIR, "experiments", "induction_suppression_bench.py"),
    "export-awq": os.path.join(BASE_DIR, "tools", "export_awq.py"),
    "train": os.path.join(BASE_DIR, "tools", "train_sft.py"),
    "telemetry": os.path.join(BASE_DIR, "tools", "forensic_api.py"),
}

def print_help():
    print("=" * 68)
    print("  QWEN3-1.7B — CONSOLA MAESTRA DE CONTROL FORENSE Y ARQUITECTURA")
    print("=" * 68)
    print("Uso: python run.py <comando>\n")
    print("Comandos disponibles:")
    print("  download        -> Descarga y verifica los pesos oficiales Safetensors (4.06 GB)")
    print("  chat            -> Inicia el chat interactivo local (soporta <think>)")
    print("  standalone-chat -> Chat nativo 100% puro PyTorch (sin runtime transformers)")
    print("  benchmark       -> Benchmark de rendimiento, latencia y VRAM (MLA, SWA, FP8)")
    print("  inspect         -> Ejecuta Rayos X del Residual Stream y generación")
    print("  test            -> Corre las 58 pruebas unitarias de la arquitectura modular")
    print("  export-gguf     -> Exportador binario a formato estándar GGUF v3 (llama.cpp / Ollama)")
    print("  export-awq      -> Exportador de cuantización 4-bit AWQ W4A16 protegiendo Dim 1793")
    print("  train           -> Pipeline de entrenamiento supervisado (SFT) y backpropagation")
    print("  telemetry       -> Servidor API de telemetría forense para Domain_Simple")
    print("  parity          -> Verificación numérica cruzada de paridad capa por capa")
    print("  induction-bench -> Banco de pruebas de inducción y tasa de supresión parásita")
    print("  refusal         -> Extracción y ablación de la dirección de rechazo r_hat")
    print("  surgery         -> Cirugía factual ROME en proyecciones MLP sin reentrenamiento")
    print("  causal          -> Ejecuta ablación de las 28 capas y trasplante causal")
    print("  induction       -> Escanea las 448 cabezas de atención buscando circuitos copiadores")
    print("  heads           -> Censo y clasificación arquetípica de las 448 cabezas")
    print("  spectral        -> Rango efectivo SVD, anisotropía y detección de dimensiones rebeldes")
    print("  suppression     -> Detección de cabezas inhibidoras (Copy-Suppression Circuits)")
    print("  dissect         -> Disección espectral (SVD) de la cabeza de inducción L16H14")
    print("  logit-lens      -> Analiza la emergencia de conceptos capa por capa")
    print("  binary          -> Inspecciona la cabecera binaria de los archivos Safetensors")
    print("  compare         -> Análisis comparativo de arquitecturas (Qwen3 vs Llama 3 vs Gemma 2)")
    print("  steer           -> Control de conceptos y steering vectors en Capa 18")
    print("  tokenizer       -> Forense exhaustivo del vocabulario y tokens especiales")
    print("=" * 68)

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help", "help"]:
        print_help()
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd not in COMMANDS:
        print(f"[-] Comando desconocido: '{cmd}'")
        print_help()
        sys.exit(1)

    target_script = COMMANDS[cmd]
    print(f"[*] Lanzando: {os.path.relpath(target_script, BASE_DIR)}\n")
    sys.stdout.flush()

    # Detectar el ejecutable que posee torch instalado
    python_bin = sys.executable
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        py312 = r"C:\Users\AltaPedagogia1\AppData\Local\Programs\Python\Python312\python.exe"
        if os.path.exists(py312):
            python_bin = py312

    subprocess.run([python_bin, target_script] + sys.argv[2:])
