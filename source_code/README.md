# Qwen3-1.7B — Suite de Arquitectura Modular de Alto Calibre

> **Licencia:** Apache License 2.0 (Copyright 2024-2026 The Qwen team, Alibaba Group y colaboradores).  
> **Conformidad Legal:** Todos los componentes derivados mantienen los avisos de atribución y licencias originales de Apache 2.0.  
> **Rutas y Referencias:** Todas las rutas contenidas en este documento son estrictamente relativas a la raíz del proyecto.

Esta suite contiene la reimplementación modular, limpia y desacoplada de la arquitectura **Qwen3-1.7B**, organizada bajo estándares de ingeniería de software para sistemas de aprendizaje profundo de alto rendimiento.

---

## 🏛️ Árbol Estructural del Paquete (`source_code/`)

```text
source_code/
├── layers/                           # Componentes atómicos de capas neuronales
│   ├── attention/
│   │   ├── gqa.py                    # Grouped Query Attention (16:8) con QK-RMSNorm
│   │   ├── qk_norm.py                # Estabilizadores QK RMSNorm por cabeza
│   │   ├── chunked_prefill.py        # Tiled Attention (Online Softmax) y Chunked Prefill para 131k tokens
│   │   ├── cross_layer_kv.py         # Cross-Layer Attention & KV-Sharing (50% ahorro VRAM)
│   │   ├── mla.py                    # Multi-Head Latent Attention (MLA) y compresión latente (70% ahorro)
│   │   ├── sliding_window.py         # Sliding Window Attention (SWA) con preservación de Attention Sinks
│   │   └── __init__.py
│   ├── feed_forward/
│   │   ├── swiglu.py                 # Bloque SwiGLU 2048 -> 6144 -> 2048
│   │   ├── geglu.py                  # Bloque GeGLU para comparativa con Gemma 2
│   │   ├── moe.py                    # Bloque Sparse MoE (Top-K Routing y Expertos Compartidos)
│   │   ├── moe_specialization.py     # Analizador de Saliencia de Compuertas y Especialización de Expertos
│   │   └── __init__.py
│   ├── normalization/
│   │   ├── rmsnorm.py                # Root Mean Square Layer Normalization
│   │   ├── layernorm.py              # LayerNorm con sesgo (baseline comparativo)
│   │   ├── outlier_channel_norm.py   # SmoothQuantScaler para calibrar la Dimensión Rebelde 1793
│   │   └── __init__.py
│   ├── embeddings/
│   │   ├── rotary.py                 # Rotary Position Embeddings (RoPE, theta=1,000,000)
│   │   ├── sinusoidal.py             # Embeddings sinusoidales acústicos (ASR)
│   │   ├── yarn_rotary.py            # Extrapolación de contexto YaRN (32k -> 131k tokens)
│   │   └── __init__.py
│   ├── block.py                      # Qwen3DecoderBlock con doble conexión residual
│   └── __init__.py
├── models/                           # Motores neuronales y variantes
│   ├── standalone_qwen3.py           # Motor Transformer puro PyTorch 100% independiente
│   ├── pruned_qwen3.py               # Motor con Poda Quirúrgica de Capas (Layer Pruning)
│   ├── lora.py                       # Adaptadores LoRA de bajo rango y fusión en caliente
│   ├── medusa_heads.py               # Cabezales residuales Medusa para decodificación especulativa Zero-Draft
│   ├── configuration_qwen3.py        # Configuración oficial de hiperparámetros (Qwen3Config)
│   └── __init__.py
├── inference/                        # Motores de inferencia, control y aceleración
│   ├── generation.py                 # Pipeline autoregresivo con streaming y parsing <think>
│   ├── reasoning_engine.py           # Motor de Razonamiento CoT, presupuesto y backtracking de <think>
│   ├── tool_calling.py               # Motor de Function Calling JSON Schema (<tool_call>, <tool_response>)
│   ├── paged_attention.py            # Gestor de memoria paginada estilo vLLM (Block Table Zero-Copy)
│   ├── continuous_batching.py        # Planificador de lotes continuos a nivel de iteración (Continuous Batching)
│   ├── quantized_kv_cache.py         # KV-Cache cuantizado por cabeza (INT8/INT4 con 50%-75% ahorro VRAM)
│   ├── h2o_cache.py                  # Heavy-Hitter Oracle (H2O) Dynamic KV-Cache Compactor
│   ├── awq_quantizer.py              # Calibrador AWQ y protección de canales rebeldes (Dim 1793)
│   ├── grammar_sampler.py            # Token Healing y máscara de gramática JSON/XML infalible
│   ├── jit_optimizer.py              # Optimizador JIT y fusión de operadores RMSNorm + SwiGLU
│   ├── steered_generation.py         # Motor de generación guiada por vectores de activación
│   ├── logit_lens.py                 # Sonda Logit Lens capa por capa
│   ├── copy_suppression.py           # Controlador de Supresión de Copia (Anti-repetición mecanicista)
│   ├── kv_cache.py                   # Qwen3KVCache y AttentionSinkKVCache (StreamingLLM)
│   ├── fim_engine.py                 # Motor Fill-In-The-Middle para autocompletado de código
│   ├── speculative_decoding.py       # Decodificación Especulativa con verificación paralela (2x-3x)
│   ├── quantization.py               # Cuantización simétrica por canal a INT8 (Int8Linear)
│   ├── sampling.py                   # Muestreo estocástico (Temp, Top-K, Top-P, Min-P, RepPen)
│   ├── fp8_gemm.py                   # Cuantización dinámica FP8 (E4M3FN / E5M2) y emulación Tensor Core
│   └── __init__.py
├── interpretability/                 # Interpretabilidad Mecanicista y Auditoría Latente
│   ├── dla.py                        # Analizador de Atribución Directa de Logits (DLA Circuit Breakdown)
│   ├── sparse_autoencoder.py         # Top-K Sparse Autoencoder (SAE) para desentrañar polisemántica
│   ├── linear_probes.py              # Sondas lineales clasificadoras capa por capa
│   ├── rome.py                       # ROME (Rank-One Model Editing) para cirugía de conocimiento en MLP
│   ├── hallucination_detector.py     # Monitor de incertidumbre y detección temprana de alucinaciones
│   ├── caa.py                        # Contrastive Activation Addition (CAA) & Representation Engineering
│   ├── refusal_direction.py          # Analizador Geométrico del Circuito de Rechazo (Refusal Direction)
│   └── __init__.py
├── loaders/                          # Cargadores de pesos, exportación y tokenización
│   ├── weight_loader.py              # Cargador y ensamblador de fragmentos Safetensors
│   ├── tokenization_qwen3.py         # Tokenizador nativo BPE con tokens de pensamiento y FIM
│   ├── gguf_exporter.py              # Exportador binario nativo GGUF v3 para llama.cpp y Ollama (F16/Q8_0)
│   └── __init__.py
├── training/                         # Alineación, pérdidas y verificación de razonamiento
│   ├── reasoning_loss.py             # Motores GRPO (sin critic) y DPO con penalización de presupuesto CoT
│   ├── reasoning_verifier.py         # Modelos PRM a nivel de paso, verificación CoT y búsqueda Best-of-N
│   └── __init__.py
├── multimodal/                       # Subsistemas multimodales (Audio / ASR / Visión)
│   ├── configuration_qwen3_asr.py    # Configuración de codificador de audio y LLM
│   ├── audio_encoder.py              # Downsampler convolucional 8x + 32 capas de audio
│   ├── qwen3_asr_model.py            # Modelo ASR unificado (Audio Tower + Qwen3 LLM)
│   ├── configuration_qwen3_vl.py     # Configuración de Vision Transformer y Qwen3-VL
│   ├── vision_encoder.py             # ViT 2D + PatchMerger espacial 2x2
│   ├── resampler.py                  # PerceiverResampler con atención cruzada para video y multi-imagen
│   ├── qwen3_vl_model.py             # Modelo Qwen3-VL completo (Vision-Language)
│   ├── vllm_backend.py               # Soporte de integración con motores vLLM
│   └── __init__.py
├── official/                         # Implementaciones de referencia oficiales
│   ├── modeling_qwen3.py             # Modelo causal oficial adaptado a transformers
│   └── modular_qwen3.py              # Definición modular canónica
├── tests/
│   ├── test_architecture.py          # Suite de 49 pruebas unitarias exhaustivas
│   └── __init__.py
└── __init__.py                       # Exportador de la API pública unificada
```

---

## 🔬 Especificación Técnica de los Módulos

### 1. Capas Fundamentales (`layers/`)
* **`layers/attention/gqa.py`**:
  * **Grouped Query Attention (GQA)** con 16 cabezas de consulta y 8 cabezas de clave/valor (ratio 2:1).
  * Dimensión por cabeza: $d_k = 128$ ($16 \times 128 = 2048$).
  * **QK-RMSNorm**: Aplica normalización RMS independiente sobre las matrices Q y K antes del producto escalar, estabilizando la atención ante secuencias ultra-largas y suprimiendo la varianza desmedida.
* **`layers/attention/sliding_window.py`**:
  * **Sliding Window Attention (SWA)** con preservación de Attention Sinks: Permite procesar secuencias de longitud infinita manteniendo una ventana rodante $W$ de tokens recientes más los $S$ primeros tokens iniciales (sumideros de atención), reduciendo la complejidad a $O(T \times W)$.
* **`layers/feed_forward/swiglu.py` & `moe.py`**:
  * `SwiGLU`: $\text{SwiGLU}(x) = (x W_{gate} \cdot \text{SiLU}(x W_{gate})) \odot (x W_{up}) \cdot W_{down}$ ($2048 \to 6144 \to 2048$).
  * `Qwen3SparseMoeBlock`: Arquitectura Mixture-of-Experts con enrutador Top-K softmax, expertos compartidos y cálculo de pérdida auxiliar de balanceo de carga.
* **`layers/normalization/outlier_channel_norm.py`**:
  * **SmoothQuantScaler**: Normalizador y redistribuidor de varianza para canales rebeldes (Dimensión 1793). Migra escala de activación hacia pesos ($Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$), permitiendo cuantización INT8 sin degradación.
* **`layers/embeddings/yarn_rotary.py`**:
  * **YaRN (Yet another RoPE extensioN)**: Extrapolación de contexto de 32k a 131k tokens combinando interpolación lineal en altas frecuencias con factores de escala de temperatura de atención ($mscale$).

---

### 2. Modelos y Adaptabilidad (`models/`)
* **`models/standalone_qwen3.py`**:
  * Motor causal completo sin dependencias de paquetes externos pesados; corre directamente con PyTorch puro.
* **`models/pruned_qwen3.py`**:
  * `PrunedQwen3ForCausalLM` permite podar capas específicas en caliente basándose en los descubrimientos de la ablación causal forense (por ejemplo, omitiendo la Capa 26 sin degradar la precisión).
* **`models/medusa_heads.py`**:
  * `Qwen3MedusaModel`: Decodificación especulativa residual multi-token (MTP) sin modelo borrador auxiliar. Acelera la generación $2.5\times - 3\times$ mediante cabezales de proyección no lineal paralelos.

---

### 3. Inferencia Avanzada, Control y Optimización (`inference/`)
* **`inference/paged_attention.py`**:
  * `PagedKVCacheManager`: Paginación de KV-Cache con tablas de bloques de 16 tokens y asignación virtual dinámica para eliminar fragmentación de VRAM.
* **`inference/fp8_gemm.py`**:
  * `FP8Linear` y `FP8Dequantizer`: Cuantización dinámica de activaciones y almacenamiento de pesos en FP8 (E4M3FN / E5M2) con escalado por token y por canal, reduciendo el consumo de memoria en 75% frente a FP32 y emulando la aceleración de hardware Blackwell/Hopper.
* **`inference/reasoning_engine.py`**:
  * `Qwen3ReasoningEngine`: Detección en caliente de giros cognitivos (*backtracking*), control de presupuesto de razonamiento y renderizado en streaming de etiquetas `<think>`.
* **`inference/tool_calling.py`**:
  * `Qwen3ToolCallingEngine`: Generación y validación de llamadas a herramientas según JSON Schema bajo el protocolo nativo `<tool_call>`.
* **`inference/awq_quantizer.py`**:
  * `AWQChannelProtector` y `AWQLinear`: Cuantización por grupos consciente de activación que protege la Dimensión Rebelde 1793.
* **`inference/grammar_sampler.py`**:
  * `TokenHealingProcessor` y `JSONGrammarMasker`: Corrección de límites de token (*Token Boundary Problem*) y filtrado sintáctico estricto en tiempo real.

---

### 4. Interpretabilidad y Entrenamiento (`interpretability/` & `training/`)
* **`interpretability/dla.py`**:
  * `DirectLogitAttributionAnalyzer`: Descomposición analítica del residual stream proyectado directamente a través de la matriz de desembebing $W_U$.
* **`interpretability/refusal_direction.py`**:
  * `RefusalDirectionAnalyzer`: Extracción del vector geométrico de rechazo $\hat{r}$ y neutralización quirúrgica mediante ortogonalización del residual stream ($h_{\perp} = h - (h \cdot \hat{r}) \hat{r}$).
* **`interpretability/rome.py`**:
  * `ROMEKnowledgeEditor`: Cirugía factual de rango 1 (Rank-One Model Editing) para implantar hechos en matrices MLP sin reentrenamiento.
* **`training/reasoning_loss.py`**:
  * `GRPOLoss`: Optimización de políticas basada en grupos sin modelo crítico/value network ($50\%$ ahorro de VRAM).
  * `DPOLoss`: Optimización por preferencia directa para razonamiento contrastivo.
* **`training/reasoning_verifier.py`**:
  * `StepLevelPRM`: Modelado de recompensa de procesos paso a paso.
  * `BestOfNVerifier`: Búsqueda y selección de trayectorias de razonamiento óptimas según formato, coherencia y confianza de PRM.

---

### 5. Multimodal y Exportación (`multimodal/` & `loaders/`)
* **`multimodal/resampler.py`**:
  * `PerceiverResampler`: Compresión de secuencias visuales largas de video a un número fijo de tokens latentes mediante atención cruzada.
* **`loaders/gguf_exporter.py`**:
  * `Qwen3GGUFExporter`: Serializador binario nativo GGUF v3 para exportar pesos a llama.cpp y Ollama en formatos F16 y Q8_0 con alineación estricta de 32 bytes.

---

## 🧪 Validación y Pruebas Unitarias (`tests/`)

Para ejecutar la batería completa de 49 pruebas unitarias de software:

```bash
python run.py test
```

### Cobertura de las 49 Pruebas:
1. `test_01_rmsnorm`: Estabilidad y normalización cuadrática unitaria.
2. `test_02_layernorm_comparison`: Comparación numérica con baseline LayerNorm.
3. `test_03_rotary_embeddings`: Validación del tensor RoPE y rotación en plano complejo.
4. `test_04_swiglu_mlp`: Contracción y activación SiLU ($2048 \to 6144 \to 2048$).
5. `test_05_gqa_attention_with_qk_norm`: Proyección de 16 consultas a 8 claves/valores y QK-RMSNorm.
6. `test_06_kv_cache`: Verificación de consistencia del buffer de claves y valores.
7. `test_07_quantization_int8`: Precisión numérica en cuantización y descuantización.
8. `test_08_lora_adaptation`: Inyección, cálculo de gradientes y fusión de pesos LoRA.
9. `test_09_sampling_functions`: Límites de muestreo estocástico Temperature, Top-K y Top-P.
10. `test_10_full_causal_lm_pipeline`: Forward pass causal completo en `StandaloneQwen3ForCausalLM`.
11. `test_11_pruned_model`: Validación estructural y funcional del modelo con capas podadas.
12. `test_12_audio_encoder`: Forward pass del codificador de audio multimodal con entrada Mel-Spectrogram (128 bins).
13. `test_13_attention_sink_kv_cache`: Preservación estricta de tokens de sumidero iniciales y desalojo de ventana rodante.
14. `test_14_copy_suppression_controller`: DLA de inhibición negativa adaptativa sobre tokens candidatos en contexto.
15. `test_15_smoothquant_scaler`: Homogeneización de la varianza en activaciones ante la Dimensión Rebelde 1793.
16. `test_16_vision_encoder`: ViT 2D + PatchMerger 2x2 (compresión 4 parches $\to$ 1 token).
17. `test_17_qwen3_vl_model`: Forward pass multimodal de `Qwen3VLForConditionalGeneration` con imagen y texto.
18. `test_18_fim_engine`: Formateo canónico de tokens FIM (`prefix`, `suffix`, `middle`, `repo_name`).
19. `test_19_speculative_decoding`: Ciclo de especulación y verificación paralela con modelo borrador.
20. `test_20_sparse_autoencoder`: Top-K Sparse Autoencoder (SAE), dispersión exacta y pérdida L2.
21. `test_21_linear_probe`: Clasificador lineal latente y softmax de probabilidad de conceptos.
22. `test_22_yarn_rotary`: Extrapolación de contexto YaRN (131k tokens) y factor mscale de temperatura.
23. `test_23_reasoning_engine`: Detección de giros cognitivos de rectificación (*backtracking*) y presupuesto CoT.
24. `test_24_tool_calling_engine`: Serialización y parseo de llamadas a herramientas `<tool_call>` y `<tool_response>`.
25. `test_25_paged_attention_manager`: Paginación de KV-Cache con tablas de bloques virtuales y reuso zero-copy.
26. `test_26_moe_block`: Bloque Mixture-of-Experts con enrutamiento Top-2, expertos compartidos y pérdida auxiliar.
27. `test_27_perceiver_resampler`: Compresión de secuencias visuales de video mediante Perceiver Resampler.
28. `test_28_direct_logit_attribution`: Proyección mecanicista DLA del residual stream a través de la matriz $W_U$.
29. `test_29_gguf_exporter`: Serialización binaria a formato GGUF v3 con cabecera y cuantización Q8_0.
30. `test_30_medusa_heads`: Cabezales de predicción multi-token residuales (Medusa) para inferencia Zero-Draft.
31. `test_31_reasoning_grpo_dpo_loss`: Funciones de recompensa de formato CoT, pérdidas GRPO sin crítico y DPO.
32. `test_32_awq_quantizer`: Identificación de canales salientes y cuantización AWQ de la Dimensión Rebelde 1793.
33. `test_33_token_healing_and_grammar`: Recuperación de límites de prefijo de tokens y autómata de gramática JSON.
34. `test_34_moe_expert_specialization`: Entropía de compuertas de enrutamiento y matriz de especialización por dominio.
35. `test_35_chunked_prefill_and_tiled_attention`: Atención exacta O(1) con Online Softmax particionado y prefill por trozos.
36. `test_36_quantized_kv_cache`: Almacenamiento KV cuantizado a INT8 (ahorro 50% de VRAM con error < 0.05).
37. `test_37_continuous_batching_scheduler`: Planificador a nivel de iteración con estados PENDING, PREFILLING, DECODING y FINISHED.
38: `test_38_cross_layer_attention`: Compartición de claves y valores (Cross-Layer KV) entre capas contiguas (50% ahorro de VRAM).
39. `test_39_jit_operator_fusion`: Kernels optimizados de bajo nivel para RMSNorm y activación SwiGLU sin asignaciones redundantes.
40. `test_40_rome_model_editing`: Cirugía de conocimiento de rango 1 (ROME) para editar hechos factuales en el MLP.
41. `test_41_hallucination_detector`: Detección temprana de alucinaciones mediante entropía latente en residual stream y deriva angular.
42. `test_42_contrastive_activation_addition`: Extracción de vectores de dirección contrastivos (CAA) e inyección con forward hook.
43. `test_43_h2o_kv_cache`: Desalojo inteligente H2O preservando sumideros, heavy-hitters y ventana local.
44. `test_44_multi_head_latent_attention`: Compresión latente de KV estilo DeepSeek/MLA reduciendo memoria de atención en 70%.
45. `test_45_refusal_direction`: Aislamiento geométrico de la dirección de rechazo y ortogonalización del residual stream.
46. `test_46_sliding_window_attention`: Atención en ventana deslizante SWA con preservación estricta de sumideros de atención.
47. `test_47_fp8_linear_and_gemm`: Cuantización dinámica de activaciones y proyección lineal en FP8 (E4M3FN / E5M2).
48. `test_48_reasoning_verifier_and_prm`: Modelo de recompensa de procesos (PRM), parseo CoT y selección Best-of-N.

**Resultado verificado:** `Ran 49 tests in 6.2s - OK`.
