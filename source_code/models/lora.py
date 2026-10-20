import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    Capa Lineal con adaptación de bajo rango (Low-Rank Adaptation - LoRA).
    Permite el ajuste fino (fine-tuning) ultra-eficiente de Qwen3 entrenando
    menos del 0.5% de los parámetros totales.
    
    Fórmula:
        h = W_0 * x + (alpha / r) * (B * A * x)
        donde A ~ N(0, 1/r), B = 0
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Matrices de bajo rango
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Congelamos los pesos base originales
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        # Inicialización de Kaiming para A, y ceros para B (empieza como función identidad)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        # Delta LoRA: (x * A^T) * B^T * scaling
        lora_out = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out

    def merge_weights(self):
        """Fusiona permanentemente los pesos de LoRA en la matriz base para inferencia rápida."""
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.data += delta_w.to(self.base_layer.weight.dtype)
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False

def apply_lora_to_model(model: nn.Module, target_modules: tuple[str, ...] = ("q_proj", "v_proj"), rank: int = 8, alpha: float = 16.0):
    """
    Inyecta adaptadores LoRA en los módulos objetivo (típicamente las proyecciones de atención).
    """
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target) and isinstance(getattr(module, target), nn.Linear):
                linear_layer = getattr(module, target)
                lora_layer = LoRALinear(linear_layer, rank=rank, alpha=alpha)
                setattr(module, target, lora_layer)
    return model
