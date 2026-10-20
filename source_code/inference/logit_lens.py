# Copyright 2025 The Qwen team, Alibaba Group and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn.functional as F

class LogitLensAnalyzer:
    """
    Analizador nativo de Logit Lens.
    Desproyecta los estados ocultos de cada una de las 28 capas Transformer
    hacia la matriz de vocabulario final (lm_head) pasando por la normalización,
    permitiendo observar en qué capa exacta 'emerge' cada concepto.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def inspect_prompt(self, prompt: str, top_k: int = 3) -> list[dict]:
        """
        Calcula la distribución de predicción de cada capa (0 a 27 + Embedding)
        para el último token del prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states # Tupla de 29 tensores

        results = []
        for layer_idx, hs in enumerate(hidden_states):
            last_token_vec = hs[:, -1, :] # [batch, hidden_dim]
            # Pasamos por la normalización final
            normed_vec = self.model.model.norm(last_token_vec)
            # Proyectamos al vocabulario
            logits = self.model.lm_head(normed_vec)[0] # [vocab_size]
            probs = F.softmax(logits, dim=-1)
            
            top_vals, top_indices = torch.topk(probs, top_k)
            candidates = [
                {"token": self.tokenizer.decode([idx.item()]).strip(), "prob": val.item()}
                for val, idx in zip(top_vals, top_indices)
            ]
            
            results.append({
                "layer_idx": layer_idx,
                "layer_name": "Embedding" if layer_idx == 0 else f"Capa {layer_idx:02d}",
                "norm": last_token_vec.norm().item(),
                "top_candidates": candidates
            })
            
        return results
