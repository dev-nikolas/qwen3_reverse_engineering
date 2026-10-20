import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))

print(f"===========================================================")
print(f"  QWEN3-1.7B - LOCAL INSPECTOR & LIVE INFERENCE")
print(f"===========================================================")
print(f"[*] Cargando modelo y pesos locales desde:")
print(f"    {MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.float32
)
model.eval()

print("[+] Modelo cargado exitosamente en memoria RAM local.")
print("[+] Total de parametros activos:", sum(p.numel() for p in model.parameters()))
print("-----------------------------------------------------------")

def inspect_prompt(prompt_text, max_new_tokens=40):
    print(f"\n[PROMPT DE ENTRADA]: \"{prompt_text}\"")
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # 1. Inspección de Activaciones Residuales Capa por Capa
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states # Tupla de 29 tensores (embedding + 28 capas)
        
        print("\n--- RAYOS X: MAGNITUD DEL RESIDUAL STREAM POR CAPA ---")
        for layer_idx in [0, 4, 8, 12, 16, 20, 24, 28]:
            norm_val = hidden_states[layer_idx][0, -1].norm().item()
            bar = "#" * int(norm_val // 2)
            layer_name = "Embedding" if layer_idx == 0 else f"Capa {layer_idx:02d}"
            print(f"{layer_name:10s} | Norma: {norm_val:6.2f} | {bar}")

    # 2. Generación en tiempo real mostrando razonamiento
    print("\n--- GENERACION EN TIEMPO REAL (TOKEN POR TOKEN) ---")
    generated_ids = input_ids[0].tolist()
    
    current_ids = input_ids
    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(current_ids)
            logits = out.logits[:, -1, :]
            
            # Top 3 candidatos para este token
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, 3)
            
            next_token_id = top_indices[0, 0].item()
            token_str = tokenizer.decode([next_token_id])
            
            # Formato de top candidatos
            cand_info = ", ".join([f"'{tokenizer.decode([idx.item()])}': {prob.item()*100:.1f}%" for idx, prob in zip(top_indices[0], top_probs[0])])
            print(f"Paso {step+1:02d} -> Emitido: {repr(token_str):12s} | Opciones: [{cand_info}]")
            
            generated_ids.append(next_token_id)
            current_ids = torch.tensor([generated_ids])
            
            if next_token_id in [tokenizer.eos_token_id, 151645]: # EOS
                print("[Fin de generacion detectado]")
                break
                
    full_text = tokenizer.decode(generated_ids)
    print("\n--- TEXTO RESULTANTE COMPLETO ---")
    print(full_text)
    print("===========================================================")

if __name__ == "__main__":
    test_prompt = "¿Cual es la capital de Francia? Pensamiento:"
    inspect_prompt(test_prompt, max_new_tokens=15)
