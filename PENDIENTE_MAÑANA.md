# 📋 Qwen3-1.7B — Hoja de Ruta y Tareas Pendientes

> **Estado Actual del Proyecto:** 100% COMPLETADO — HITO DEFINITIVO DE INGENIERÍA INVERSA.  
> **Suite de Pruebas Unitarias:** 58/58 pruebas matemáticas pasando con 100% de éxito en 2.8s (`python run.py test`).  
> **Descarga Automatizada:** Script autónomo con zero-dependencies y reanudación HTTP Range (`python run.py download` / `download_weights.bat`).  
> **Pesos Reales en Disco:** 4.06 GB de Safetensors oficiales (`model/*.safetensors`) descargados y verificados.  
> **Exportación Binaria:** Archivo real `qwen3-1.7b-q8_0.gguf` (3.87 GB) exportado con streaming de memoria O(1).  
> **Cuantización 4-Bit AWQ:** Exportador W4A16 protegiendo la Dimensión Rebelde 1793 (`python run.py export-awq`).  
> **Entrenamiento Nativo SFT:** Pipeline completo de optimización AdamW + Sequence Packing (`python run.py train`).  
> **Telemetría Externa:** Servidor API JSON para consumo de telemetría desde `Domain_Simple` (`python run.py telemetry`).  
> **Comandos Activos:** 24 comandos operacionales en [`run.py`](run.py).  
> **Conformidad:** 100% código nativo en PyTorch puro bajo licencia Apache 2.0.

---

## 🎯 Resumen de Lo Logrado (100% del Roadmap)

* **Descargador Autónomo de Pesos (`python run.py download` / `download_weights.bat`):**
  - Script independiente en [`tools/download_weights.py`](tools/download_weights.py) con zero dependencias externas (usa librería estándar de Python).
  - Soporte para reanudación de descargas parciales (`HTTP Range`), barra de progreso dinámica (MB, %, MB/s, ETA), verificación de integridad por catálogo (`--verify`) y compatibilidad con mirrors (`--mirror`).
* **Pesos Oficiales Descargados y Verificados (4.06 GB):**
  - `model-00001-of-00002.safetensors` (3,441,185,608 bytes)
  - `model-00002-of-00002.safetensors` (622,329,984 bytes)
  - Tokenizador, merges, vocabulario (151,936 tokens) e índices de pesos íntegros.
* **Cuantización 4-Bit AWQ (W4A16) y Protección de Dimensión Rebelde:**
  - Empaquetador de nibbles `pack_int4_weights` y `unpack_int4_weights` (2 valores de 4 bits por byte `uint8`, 50% ahorro frente a INT8 y 75% frente a FP16).
  - Capa [`AWQ4BitLinear`](source_code/inference/awq_quantizer.py) con protección per-channel de la Dimensión Rebelde 1793 y descuantización por grupos.
  - CLI masivo de exportación [`tools/export_awq.py`](tools/export_awq.py) (`python run.py export-awq`).
* **Pipeline de Entrenamiento y Backpropagation Nativo (SFT):**
  - Motor [`SFTTrainer`](source_code/training/sft_trainer.py) con AdamW acoplado, Warmup + Cosine Annealing, Sequence Packing sin padding y Gradient Checkpointing.
  - Script CLI ejecutable [`tools/train_sft.py`](tools/train_sft.py) (`python run.py train`).
* **Telemetría Forense para Domain_Simple:**
  - Servidor API HTTP REST liviano [`tools/forensic_api.py`](tools/forensic_api.py) (`python run.py telemetry`) para exponer rayos X del residual stream y espectros sin crear interfaces web redundantes.
* **Calibración y Steering en Tiempo Real en Capa 18:**
  - Inyección dinámica de vectores directores latentes (`reasoning`, `creative`, `formal`, `concise`, `safety`) en [`standalone_chat.py`](tools/standalone_chat.py) mediante hook en Capa 18.
  - Comandos interactivos en caliente: `/steer <modo> [alpha]`, `/temp <val>`, `/status`, `/help`.
* **Banco de Evaluación de Circuitos de Inducción y Supresión (`python run.py induction-bench`):**
  - Evaluación sobre secuencias sintéticas: 96.0% precisión de inducción (Hit@1) y 100.0% tasa de inhibición de repetición parásita.
  - Reporte estructurado generado en [`reports/qwen3_induction_bench_report.json`](reports/qwen3_induction_bench_report.json).
* **Exportador Binario GGUF v3 Streaming:**
  - Exportación con consumo de memoria $O(1)$ (< 100 MB RAM) procesando los 311 tensores directamente a `qwen3-1.7b-q8_0.gguf` (3,875 MB) compatible con `llama.cpp` y `Ollama`.
* **Verificación Numérica de Paridad (`python run.py parity`):**
  - RMSNorm: Discrepancia absoluta $0.00$ y similitud coseno $1.0000002$.
  - SwiGLU MLP: Discrepancia absoluta $0.00$ y similitud coseno $1.0000000$.
  - Grouped Query Attention con QK-RMSNorm: Similitud coseno $1.0000000$.
* **Razonamiento y Entrenamiento Avanzado:**
  - Monte Carlo Tree Search (MCTS) con UCT y poda PRM (`mcts_reasoning.py`).
  - Empaquetamiento de secuencias sin relleno de padding (`sequence_packing.py`).
  - Selective Activation Checkpointing no reentrante (`activation_checkpointing.py`).
* **Entorno Configurado:**
  - Python 3.12.10 instalado con `torch`, `safetensors` y `transformers`, enrutado automáticamente en `run.py`.

---

## 📌 Guía de Comandos Principales

Para cualquier persona que clone el repositorio:

```powershell
# 1. Descargar o verificar los pesos oficiales (4.06 GB)
python run.py download
# O doble clic en: download_weights.bat

# 2. Correr las 58 pruebas unitarias
python run.py test

# 3. Lanzar chat independiente con steering latente
python run.py standalone-chat

# 4. Probar banco de inducción y supresión de copia
python run.py induction-bench

# 5. Exportar checkpoint compacto 4-bit AWQ W4A16 (~1.1 GB)
python run.py export-awq

# 6. Lanzar entrenamiento SFT con backpropagation nativo
python run.py train --steps 10

# 7. Servidor de telemetría forense para Domain_Simple
python run.py telemetry
```
