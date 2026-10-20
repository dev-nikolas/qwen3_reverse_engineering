import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))

print("=" * 70)
print("  REPRESENTATION ENGINEERING: VECTORES DE CONDUCCIÓN (STEERING VECTORS)")
print("=" * 70)
print("[*] Cargando modelo...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
model.eval()

# 1. Extracción del Vector Director (Steering Vector)
# Contrastamos dos estados mentales:
# Positivo: "Razonamiento logico, cientifico, detallado y paso a paso."
# Negativo: "Respuesta impulsiva, rapida, superficial y sin pensar."

prompt_pos = "Razonamiento logico, cientifico, formal, detallado y reflexivo."
prompt_neg = "Respuesta impulsiva, informal, descuidada, rapida y sin pensar."

inputs_pos = tokenizer(prompt_pos, return_tensors="pt")
inputs_neg = tokenizer(prompt_neg, return_tensors="pt")

target_layer = 18 # Nuestra capa de control central descubierta en el trasplante causal

with torch.no_grad():
    out_pos = model(**inputs_pos, output_hidden_states=True)
    out_neg = model(**inputs_neg, output_hidden_states=True)

h_pos = out_pos.hidden_states[target_layer][0, -1, :] # [2048]
h_neg = out_neg.hidden_states[target_layer][0, -1, :] # [2048]

# El vector director es la diferencia en el espacio de activaciones
steering_vector = h_pos - h_neg
steering_vector = steering_vector / steering_vector.norm() # Normalizado

print(f"[+] Vector director de formalidad y razonamiento extraído en Capa {target_layer}.")
print(f"    Norma unitaria: {steering_vector.norm().item():.4f} | Dimension: {steering_vector.shape[0]}")

# 2. Prueba de Intervención: Evaluar la misma pregunta bajo 3 condiciones mentales:
# A) Normal (sin alteración)
# B) Conducción Positiva (+ Formalidad/Ciencia: +alpha * vector)
# C) Conducción Negativa (- Formalidad/Ciencia: -alpha * vector)

test_query = "¿Que es el tiempo?"
test_inputs = tokenizer(test_query, return_tensors="pt")

def generate_steered(alpha: float, max_tokens: int = 15):
    # Hook en la capa 18 para sumar alpha * steering_vector en cada paso
    def steering_hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        tensor_out = output[0] if is_tuple else output
        modified = tensor_out.clone()
        modified[:, -1, :] += alpha * steering_vector
        return (modified,) + output[1:] if is_tuple else modified

    handle = model.model.layers[target_layer].register_forward_hook(steering_hook)
    
    current_ids = test_inputs["input_ids"]
    generated_ids = current_ids[0].tolist()
    
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(current_ids).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).item()
            if next_token in [tokenizer.eos_token_id, 151645]:
                break
            generated_ids.append(next_token)
            current_ids = torch.tensor([generated_ids])
            
    handle.remove()
    return tokenizer.decode(generated_ids)

print("\n--- COMPARATIVA DE INFLUENCIA EN EL PENSAMIENTO ---")
print(f"Pregunta base: \"{test_query}\"")

text_normal = generate_steered(alpha=0.0)
print(f"\n[A] Estado Normal (alpha = 0.0):\n    -> {text_normal}")

text_pos = generate_steered(alpha=15.0)
print(f"\n[B] Conducción Positiva (+15.0 Vector Razonamiento Científico):\n    -> {text_pos}")

text_neg = generate_steered(alpha=-15.0)
print(f"\n[C] Conducción Negativa (-15.0 Inversión de Razonamiento):\n    -> {text_neg}")

print("\n" + "=" * 70)
print("  EXPERIMENTO DE CONDUCCIÓN MENTAL COMPLETADO")
print("=" * 70)
