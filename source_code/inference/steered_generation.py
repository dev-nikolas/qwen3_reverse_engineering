# Copyright 2025 The Qwen team, Alibaba Group and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

import torch
from .sampling import sample_next_token, apply_repetition_penalty

class SteeredGenerator:
    """
    Motor de generación con inyección de Vectores Directores (Steering Vectors).
    Permite modular la respuesta del modelo en tiempo real (más analítico, formal,
    creativo o conciso) mediante adición directa de vectores en la Capa 18 sin reentrenar.
    """
    def __init__(self, model, tokenizer, target_layer: int = 18, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.device = device

    def generate_steered(
        self,
        prompt: str,
        steering_vector: torch.Tensor | None = None,
        alpha: float = 0.0,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.15
    ) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        generated = input_ids[0].tolist()
        new_tokens = []

        hook_handle = None
        if steering_vector is not None and alpha != 0.0:
            steer_vec = steering_vector.to(self.device)
            if steer_vec.norm() > 0:
                steer_vec = steer_vec / steer_vec.norm()

            def hook_fn(module, args, output):
                is_tuple = isinstance(output, tuple)
                tensor_out = output[0] if is_tuple else output
                steered = tensor_out.clone()
                steered[:, -1, :] += alpha * steer_vec
                return (steered,) + output[1:] if is_tuple else steered

            hook_handle = self.model.model.layers[self.target_layer].register_forward_hook(hook_fn)

        current_ids = input_ids
        try:
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = self.model(current_ids).logits[:, -1, :]
                    logits = apply_repetition_penalty(logits, new_tokens, penalty=repetition_penalty)
                    next_token = sample_next_token(logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    
                    if next_token in [self.tokenizer.eos_token_id, 151645, 151643]:
                        break
                        
                    generated.append(next_token)
                    new_tokens.append(next_token)
                    current_ids = torch.tensor([generated], device=self.device)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        return self.tokenizer.decode(new_tokens)
