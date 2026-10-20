import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sampling import sample_next_token, apply_repetition_penalty

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))

print("=" * 65)
print("  QWEN3-1.7B — CONSOLA DE CHAT INTERACTIVO (100% LOCAL)")
print("=" * 65)
print(f"[*] Cargando pesos desde: {MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
model.eval()

print("[+] Modelo cargado en RAM. ¡Listo para conversar!")
print("[*] Escribe 'salir' o 'exit' para terminar.")
print("-" * 65)

conversation_history = []

def chat():
    while True:
        try:
            user_input = input("\n[Tú] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("\n[+] Sesión finalizada.")
                break
                
            # Formateamos con template de Chat
            messages = conversation_history + [{"role": "user", "content": user_input}]
            
            try:
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted_prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
                
            input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"]
            generated_ids = input_ids[0].tolist()
            
            print("\n[Qwen3-1.7B] > ", end="", flush=True)
            
            in_thinking_mode = False
            current_ids = input_ids
            max_new_tokens = 256
            
            new_tokens_generated = []
            
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = model(current_ids).logits[:, -1, :]
                    
                    # Penalización de repetición
                    logits = apply_repetition_penalty(logits, new_tokens_generated, penalty=1.15)
                    
                    # Muestreo
                    next_token_id = sample_next_token(logits, temperature=0.6, top_p=0.9, top_k=40)
                    
                    if next_token_id in [tokenizer.eos_token_id, 151645, 151643]:
                        break
                        
                    token_str = tokenizer.decode([next_token_id])
                    
                    # Detectamos bloque <think>
                    if next_token_id == 151667 or "<think>" in token_str:
                        in_thinking_mode = True
                        print("\n[Pensamiento]: ", end="", flush=True)
                    elif next_token_id == 151668 or "</think>" in token_str:
                        in_thinking_mode = False
                        print("\n[Respuesta]: ", end="", flush=True)
                    else:
                        print(token_str, end="", flush=True)
                        
                    generated_ids.append(next_token_id)
                    new_tokens_generated.append(next_token_id)
                    current_ids = torch.tensor([generated_ids])
                    
            print()
            assistant_reply = tokenizer.decode(new_tokens_generated).replace("<think>", "").replace("</think>", "").strip()
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            
        except KeyboardInterrupt:
            print("\nInterrupción detectada. Saliendo...")
            break

if __name__ == "__main__":
    chat()
