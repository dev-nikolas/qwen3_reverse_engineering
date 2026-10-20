import torch
import torch.nn.functional as F

def apply_repetition_penalty(logits: torch.Tensor, generated_tokens: list[int], penalty: float = 1.1) -> torch.Tensor:
    """
    Penaliza los logits de los tokens que ya han sido emitidos previamente
    para evitar bucles repetitivos infinitos.
    """
    if penalty == 1.0 or not generated_tokens:
        return logits
    
    unique_tokens = list(set(generated_tokens))
    for token_id in unique_tokens:
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    min_p: float = 0.05
) -> int:
    """
    Aplica muestreo estocástico avanzado (Temperature, Top-K, Nucleus Top-P y Min-P).
    Si temperature <= 0.0, aplica Greedy Search determinista (argmax).
    """
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1).item()

    # 1. Escalamiento por temperatura
    logits = logits / max(temperature, 1e-5)

    # 2. Filtrado Top-K
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    # 3. Filtrado Top-P (Nucleus)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = -float("Inf")

    # 4. Filtrado Min-P (Elimina tokens cuya prob sea menor que min_p * prob_maxima)
    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        max_prob = torch.max(probs)
        min_p_threshold = max_prob * min_p
        logits[probs < min_p_threshold] = -float("Inf")

    # 5. Muestreo multinomial
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()
