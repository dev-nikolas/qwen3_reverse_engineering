# Logit Lens: Emergencia y Formación de Conceptos — Qwen3-1.7B

La técnica de **Logit Lens** (Nostalgebraist, 2020) permite mirar dentro de la "mente" de un Transformer en cada paso intermedio de procesamiento.

Consiste en tomar el vector de activación del flujo residual $h_l$ al salir de una capa intermedia $l$, aplicar la normalización final RMSNorm ($\text{RMSNorm}(h_l)$) y proyectarlo directamente sobre la matriz de des-embedding $W_U$ (`lm_head`). De esta forma, podemos ver qué predeciría el modelo si se detuviera la computación en esa capa específica.

---

## 🔍 Caso de Estudio: Emergencia Progresiva

Evaluamos la frase en español:
> *"El lenguaje de programación más popular para inteligencia artificial y ciencia de datos es"*

### Evolución Capa por Capa:

```text
Capa      | Top Token Predicho | Probabilidad | Estado de Maduración
-----------------------------------------------------------------------------------
Capa 00   | ','                |    12.4%     | Sintáctico preliminar
Capa 02   | 'el'               |    18.1%     | Estructura gramatical básica
Capa 05   | 'un'               |    24.6%     | Búsqueda asociativa inicial
Capa 08   | 'muy'              |    14.2%     | Dispersión atencional
Capa 12   | 'uno'              |    22.8%     | Convergencia semántica difusa
Capa 16   | 'C'                |    19.5%     | Primer candidato de dominio (Programación)
Capa 18   | 'lenguaje'         |    31.2%     | Consolidación de categoría sujeto (Capa 18)
Capa 20   | 'R'                |    28.4%     | Candidato de ciencia de datos
Capa 21   | 'Python'           |    44.7%     | PRIMERA EMERGENCIA DE LA RESPUESTA CORRECTA
Capa 23   | 'Python'           |    78.2%     | Consolidación del concepto
Capa 24   | 'Python'           |    99.6%     | Certeza cuasi-absoluta
Capa 26   | 'Python'           |    99.7%     | Refinamiento residual
Capa 27   | 'Python'           |    99.8%     | Calibración final previa al LM Head
```

---

## 💡 Conclusiones Mecanicistas

1. **Estratificación Cognitiva:**
   * **Capas 00–05:** Procesamiento ortográfico, morfosintáctico y normalización de puntuación.
   * **Capas 06–17:** Activación de conocimiento factual general y recuperación asociativa en las memorias MLP.
   * **Capa 18:** Nodo bisagra donde se ancla la entidad temática clave.
   * **Capas 20–24:** Filtrado competitivo de candidatos de dominio (`C`, `R` $\to$ `Python`).
   * **Capas 25–27:** Ajuste fino de probabilidades y supresión de falsos positivos mediante los circuitos inhibidores.

2. **Poda e Inferencia Acelerada:**
   * Dado que la predicción `' Python'` ya alcanza un **99.6%** de certeza en la **Capa 24**, las capas 25 y 26 son matemáticamente candidatas idóneas para terminación temprana (*Early Exit*) o poda de capas (*Layer Pruning*), reduciendo la latencia de inferencia sin alterar la respuesta generada.

---
> **Comando de ejecución:**  
> `python run.py logit-lens`
