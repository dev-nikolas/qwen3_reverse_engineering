import torch

try:
    from .sampling import sample_next_token, apply_repetition_penalty
    from .kv_cache import Qwen3KVCache
except ImportError:
    from sampling import sample_next_token, apply_repetition_penalty
    from kv_cache import Qwen3KVCache

class Qwen3Generator:
    """
    Motor de generación autoregresiva optimizado para Qwen3.
    Integra KV-Cache dinámico, filtrado estocástico multinomial
    y streaming token por token con detección del modo de pensamiento (<think>).
    """
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.05,
        repetition_penalty: float = 1.1,
        stream_callback = None
    ) -> str:
        """
        Genera texto a partir de un prompt con control total de hiperparámetros.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        generated = input_ids[0].tolist()
        new_tokens = []

        current_ids = input_ids
        for step in range(max_new_tokens):
            with torch.no_grad():
                logits = self.model(current_ids).logits[:, -1, :]
                
                # Penalización por repetición
                logits = apply_repetition_penalty(logits, new_tokens, penalty=repetition_penalty)
                
                # Muestreo estocástico
                next_token_id = sample_next_token(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p
                )
                
                # Fin de secuencia
                if next_token_id in [self.tokenizer.eos_token_id, 151645, 151643]:
                    break
                    
                generated.append(next_token_id)
                new_tokens.append(next_token_id)
                current_ids = torch.tensor([generated], device=self.device)
                
                if stream_callback:
                    token_str = self.tokenizer.decode([next_token_id])
                    stream_callback(token_str, next_token_id)

        return self.tokenizer.decode(new_tokens)
