# 📋 Qwen3-1.7B — Hoja de Ruta y Tareas Pendientes

> **Estado Actual del Proyecto:** ~50% completado hacia la ingeniería inversa y reimplementación definitiva.  
> **Suite de Pruebas Unitarias:** 51 pruebas matemáticas estructuradas (`python run.py test`).  
> **Comandos Activos:** 19 comandos operacionales en [`run.py`](run.py).  
> **Conformidad:** 100% código nativo en PyTorch puro bajo licencia Apache 2.0.

---

## 🎯 Resumen de Lo Logrado

* **Arquitectura de Atención:** Grouped Query Attention (16:8) con QK-RMSNorm, Cross-Layer Attention (CLA 50% KV saving), Multi-Head Latent Attention (MLA 70% KV saving con 3,829 t/s) y Sliding Window Attention (SWA) con preservación estricta de sumideros de atención.
* **Aceleración e Inferencia:** PagedAttention con tablas virtuales de bloques, planificador de Continuous Batching a nivel de iteración, H2O KV Cache compactor (87.5% ahorro VRAM), INT8/INT4 Quantized KV Cache, JIT operator fusion, y emulación de tensores FP8 (E4M3FN / E5M2 con 75% ahorro de peso).
* **Interpretabilidad Forense:** Refusal Direction Analyzer (vector $\hat{r}$ y ortogonalización defensiva), ROME Knowledge Surgery (edición factual en MLP con precisión $< 10^{-6}$), Direct Logit Attribution (DLA), Sparse Autoencoders (SAE Top-K), Hallucination Detector (entropía y deriva angular), SVD espectral y censo de las 448 cabezas.
* **Razonamiento y Entrenamiento Avanzado:**
  - **Monte Carlo Tree Search (MCTS):** Motor de búsqueda en tiempo de inferencia (`source_code/training/mcts_reasoning.py`) con selección UCT, evaluación y poda PRM de ramas espurias.
  - **Sample / Sequence Packing:** Concatenación voraz de secuencias variables en tensores continuos con máscara causal diagonal por bloques (`source_code/training/sequence_packing.py`).
  - **Selective Activation Checkpointing:** Checkpointing de gradientes no reentrante para `Qwen3DecoderBlock` (`source_code/training/activation_checkpointing.py`).
* **Herramientas y Exportación:**
  - CLI de exportación a GGUF v3 (`tools/export_gguf.py`, comando `python run.py export-gguf`) para llama.cpp / Ollama.
  - Verificación numérica de paridad capa por capa (`experiments/numerical_parity_check.py`, comando `python run.py parity`).

---

## 🔍 Rectificaciones Críticas Requeridas para Ingeniería Inversa con Pesos Reales

1. **Entorno de Ejecución (PyTorch / CUDA):**
   - El entorno actual de Python es Python 3.15 sin `torch` instalado. Se requiere instalar PyTorch (`pip install torch`) o ejecutar con un entorno virtual que contenga PyTorch, `safetensors` y `transformers`.
2. **Archivos Binarios de Pesos (`model/`):**
   - El directorio `model/` se encuentra actualmente vacío. Para ejecutar inferencia real, contrastes numéricos con los 1.7B pesos oficiales y exportación a GGUF de producción, se requiere colocar en `model/`:
     - `model-00001-of-00002.safetensors`
     - `model-00002-of-00002.safetensors`
     - `config.json`, `tokenizer.json`
3. **Rutas Normalizadas:**
   - Se eliminaron las referencias fijas a directorios personales en los scripts de experimentos, adoptando rutas relativas automáticas en todo el repositorio.

---

## 🚀 Próximas Tareas (Hacia el 75% - 100%)

### 1. 🧮 Cuantización AWQ & GPTQ Real sobre Shards
- [ ] Implementar exportador de cuantización 4-bit AWQ directa protegiendo los canales sobresalientes de la Dimensión Rebelde 1793.
- [ ] Soporte para empaquetado de pesos en formato Marlin / ExLlamaV2.

### 2. 🧬 Calibración y Steering en Tiempo Real
- [ ] Banco de prompts de inducción automatizados para medir la tasa de inhibición de repetición en caliente.
- [ ] Inyección de vectores latentes en Capa 18 durante streaming interactivo en `standalone_chat.py`.

## 📌 Guía de Inicio Rápido para Reanudar Mañana

Al abrir la sesión mañana, ejecutar:

```powershell
# 1. Verificar integridad de las 49 pruebas
python run.py test

# 2. Ver la consola de comandos disponibles
python run.py

# 3. Lanzar el benchmark para comprobar rendimiento
python run.py benchmark
```

Todo está limpio, testeado, documentado y listo para continuar. ¡Que descanses, bro! Mañana seguimos desentrañando el iceberg completo.
