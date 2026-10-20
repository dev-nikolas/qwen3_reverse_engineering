import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))

print("=" * 65)
print("  QWEN3-1.7B - DEEP BEHAVIORAL & CAUSAL REVERSE ENGINEERING")
print("=" * 65)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
model.eval()

# ==============================================================================
# EXPERIMENTO 1: MAPA DE VULNERABILIDAD POR ABLACIÓN DE CAPAS (LAYER KNOCKOUT)
# ==============================================================================
# Si desconectamos completamente una capa (bypass directo del residual stream),
# ¿cuánto se destruye la coherencia del modelo?
# Esto revela qué capas son órganos vitales vs capas redundantes.
# ==============================================================================
print("\n" + "=" * 65)
print("  EXPERIMENTO 1: MAPA DE CRITICIDAD POR ABLACIÓN (LAYER KNOCKOUT)")
print("=" * 65)
print("[*] Desconectando quirúrgicamente una capa a la vez (0 a 27)...")

prompt_test = "El agua hierve a cien grados Celsius y se convierte en"
inputs = tokenizer(prompt_test, return_tensors="pt")
target_token_str = " vapor"
target_token_id = tokenizer.encode(target_token_str, add_special_tokens=False)[0]

# 1. Baseline sin intervención
with torch.no_grad():
    base_logits = model(**inputs).logits[0, -1, :]
    base_prob = F.softmax(base_logits, dim=-1)[target_token_id].item()
    top_base = tokenizer.decode([torch.argmax(base_logits).item()])

print(f"[Baseline] Prediccion normal: '{top_base.strip()}' con P({target_token_str.strip()}) = {base_prob*100:.2f}%\n")

layer_impacts = []
total_layers = len(model.model.layers)

for l_idx in range(total_layers):
    orig_layer = model.model.layers[l_idx]
    
    # Creamos un bypass idéntico: la capa no hace nada, solo retorna el input
    class IdentityLayer(torch.nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states
            
    model.model.layers[l_idx] = IdentityLayer()
    
    with torch.no_grad():
        ablated_logits = model(**inputs).logits[0, -1, :]
        ablated_prob = F.softmax(ablated_logits, dim=-1)[target_token_id].item()
        ablated_top = tokenizer.decode([torch.argmax(ablated_logits).item()])
        
        # Pérdida de probabilidad del token correcto
        drop = base_prob - ablated_prob
        layer_impacts.append((l_idx, ablated_top.strip(), ablated_prob, drop))
        
    # Restauramos la capa original inmediatamente
    model.model.layers[l_idx] = orig_layer

# Mostramos los resultados
print(f"{'Capa':<6} | {'Top Prediccion':<16} | {'P(target)':<10} | {'Impacto / Caída':<18} | {'Nivel de Criticidad'}")
print("-" * 75)
for l_idx, top_word, p_val, drop in layer_impacts:
    bar_len = int(max(0, drop / base_prob) * 20)
    bar = "X" * bar_len
    if drop > 0.5 * base_prob:
        crit = "CRÍTICA (Órgano Vital)"
    elif drop > 0.1 * base_prob:
        crit = "IMPORTANTE"
    elif drop <= 0:
        crit = "REDUNDANTE (Mejora o Neutra)"
    else:
        crit = "LEVE (Refinamiento)"
    print(f"Capa {l_idx:02d} | '{top_word}':{p_val*100:6.1f}% | Caída: {drop*100:5.1f}% | {bar:<20} | {crit}")

# ==============================================================================
# EXPERIMENTO 2: ACTIVATION PATCHING (CONTROL CAUSAL MENTE-A-MENTE)
# ==============================================================================
# Demostramos que podemos cambiar la decisión de Qwen3 trasplantando
# el estado oculto de una frase ("Roma") hacia otra ("París").
# ==============================================================================
print("\n" + "=" * 65)
print("  EXPERIMENTO 2: ACTIVATION PATCHING (TRASPLANTE CAUSAL DE DECISIÓN)")
print("=" * 65)

clean_prompt = "El monumento mas iconico de Francia es la Torre"  # Espera " Eiffel"
corrupt_prompt = "El monumento mas iconico de Italia es el Col"     # Espera "iseo"

clean_in = tokenizer(clean_prompt, return_tensors="pt")
corrupt_in = tokenizer(corrupt_prompt, return_tensors="pt")

with torch.no_grad():
    clean_out = model(**clean_in, output_hidden_states=True)
    corrupt_out = model(**corrupt_in, output_hidden_states=True)

clean_token = tokenizer.decode([torch.argmax(clean_out.logits[0, -1, :]).item()]).strip()
corrupt_token = tokenizer.decode([torch.argmax(corrupt_out.logits[0, -1, :]).item()]).strip()

print(f"Frase A (Francia): predice '{clean_token}'")
print(f"Frase B (Italia) : predice '{corrupt_token}'")
print("\n[*] Trasplantando el estado de la Capa de Frase B hacia Frase A en capas intermedias...")

# Probamos trasplantar el vector residual en capas 14, 18, 22
for patch_layer in [10, 14, 18, 20, 22, 24]:
    # Hook para inyectar la activacion de B en A
    def patch_hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        tensor_out = output[0] if is_tuple else output
        h = tensor_out.clone()
        h[0, -1, :] = corrupt_out.hidden_states[patch_layer + 1][0, -1, :]
        return (h,) + output[1:] if is_tuple else h

    handle = model.model.layers[patch_layer].register_forward_hook(patch_hook)
    with torch.no_grad():
        patched_logits = model(**clean_in).logits[0, -1, :]
        patched_top_id = torch.argmax(patched_logits).item()
        patched_top = tokenizer.decode([patched_top_id]).strip()
        probs = F.softmax(patched_logits, dim=-1)
        prob_eiffel = probs[tokenizer.encode(" Eiffel", add_special_tokens=False)[0]].item() * 100
        prob_coliseo = probs[tokenizer.encode("iseo", add_special_tokens=False)[0]].item() * 100
        
    handle.remove()
    print(f"Parche en Capa {patch_layer:02d} -> Nueva Prediccion: '{patched_top}' | P(Eiffel): {prob_eiffel:.1f}% | P(Coliseo): {prob_coliseo:.1f}%")

# ==============================================================================
# EXPERIMENTO 3: RADIOGRAFÍA DE NEURONAS DE CONOCIMIENTO EN EL MLP
# ==============================================================================
# Encontramos cuáles de las 6144 neuronas de la capa crítica se activan
# más fuertemente para un concepto y decodificamos su vocabulario asociado.
# ==============================================================================
print("\n" + "=" * 65)
print("  EXPERIMENTO 3: DETECCIÓN DE NEURONAS DE MEMORIA (MLP 6,144D)")
print("=" * 65)

# Tomamos la capa 18 (capa de memoria media)
layer_idx = 18
mlp = model.model.layers[layer_idx].mlp

test_phrase = "Python es un lenguaje de programacion orientado a objetos"
inputs_phrase = tokenizer(test_phrase, return_tensors="pt")

# Capturamos la activacion intermedia del MLP: act_fn(gate_proj(x)) * up_proj(x)
with torch.no_grad():
    hs = model(**inputs_phrase, output_hidden_states=True).hidden_states[layer_idx]
    norm_hs = model.model.layers[layer_idx].post_attention_layernorm(hs)
    # Activacion intermedia (dimension 6144)
    intermediate = F.silu(mlp.gate_proj(norm_hs)) * mlp.up_proj(norm_hs)
    # Tomamos el ultimo token
    activations = intermediate[0, -1, :] # [6144]
    
# Top 5 neuronas mas encendidas
top_vals, top_neurons = torch.topk(activations, 5)
print(f"Frase evaluada: \"{test_phrase}\"")
print(f"Top 5 neuronas mas encendidas en la Capa {layer_idx} (de 6,144 neuronas totales):")

for rank, (val, n_idx) in enumerate(zip(top_vals, top_neurons)):
    n_id = n_idx.item()
    # Que proyecta esta neurona de vuelta al residual stream?
    # Su peso en down_proj es la columna n_id: down_proj.weight[:, n_id]
    neuron_vector = mlp.down_proj.weight[:, n_id] # [2048]
    
    # Proyectamos este vector al vocabulario para ver que palabras 'escribe' en la mente del modelo
    normed_v = model.model.norm(neuron_vector.unsqueeze(0))
    vocab_logits = model.lm_head(normed_v)[0]
    top_vocab_ids = torch.topk(vocab_logits, 3).indices
    associated_words = [tokenizer.decode([idx.item()]).strip() for idx in top_vocab_ids]
    
    print(f"  #{rank+1} Neurona ID {n_id:4d} | Activacion: {val.item():6.2f} | Palabras que empuja: {associated_words}")

print("\n" + "=" * 65)
print("  ANALISIS CAUSAL Y FORENSE COMPLETADO")
print("=" * 65)
