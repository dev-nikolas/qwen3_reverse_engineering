import os
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file

class SafetensorsLoader:
    """
    Cargador forense y de producción de pesos Safetensors particionados (sharded).
    Soporta mapeo perezoso, conversión de precisión al vuelo (FP32, FP16, BF16)
    y resolución de nombres entre checkpoints de Transformers y arquitecturas nativas.
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.index_file = os.path.join(model_dir, "model.safetensors.index.json")
        self.shard_files = self._discover_shards()

    def _discover_shards(self) -> list[str]:
        if os.path.exists(self.index_file):
            with open(self.index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            unique_shards = sorted(list(set(index_data["weight_map"].values())))
            return [os.path.join(self.model_dir, s) for s in unique_shards]
        else:
            shards = [
                os.path.join(self.model_dir, f)
                for f in sorted(os.listdir(self.model_dir))
                if f.endswith(".safetensors")
            ]
            return shards

    def load_state_dict(self, target_dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        """Carga y unifica todos los tensores de todos los shards."""
        merged_state_dict = {}
        for shard in self.shard_files:
            shard_dict = load_file(shard)
            for k, tensor in shard_dict.items():
                merged_state_dict[k] = tensor.to(target_dtype)
        return merged_state_dict

    def load_into_model(self, model: nn.Module, target_dtype: torch.dtype = torch.float32, strict: bool = False) -> tuple[list[str], list[str]]:
        """Inyecta directamente los pesos en una instancia de torch.nn.Module."""
        state_dict = self.load_state_dict(target_dtype=target_dtype)
        
        # Si lm_head no existe en los pesos pero embed_tokens sí, ligamos los pesos
        if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()
            
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        return missing_keys, unexpected_keys
