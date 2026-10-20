import torch
import torch.nn as nn

class Int8Linear(nn.Module):
    """
    Capa Lineal con cuantización de pesos por canal a INT8 (Weight-Only Quantization).
    Reduce el consumo de memoria RAM de los pesos en un 50% frente a FP16 y un 75% frente a FP32.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Pesos almacenados en int8 [-128, 127]
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        # Escala por fila (por canal de salida) en float32
        self.register_buffer("scales", torch.zeros((out_features, 1), dtype=torch.float32))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, linear_module: nn.Linear) -> "Int8Linear":
        """Convierte una capa nn.Linear float estándar en Int8Linear simétrica."""
        q_layer = cls(linear_module.in_features, linear_module.out_features, bias=linear_module.bias is not None)
        w = linear_module.weight.data.float()
        
        # Escala máxima absoluta por fila
        max_val = torch.max(torch.abs(w), dim=1, keepdim=True)[0].clamp(min=1e-5)
        scales = max_val / 127.0
        
        # Cuantización simétrica
        w_int8 = torch.round(w / scales).clamp(-128, 127).to(torch.int8)
        
        q_layer.weight_int8.copy_(w_int8)
        q_layer.scales.copy_(scales)
        if linear_module.bias is not None:
            q_layer.bias.data.copy_(linear_module.bias.data)
            
        return q_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Descuantización al vuelo durante el forward pass
        w_dequant = (self.weight_int8.float() * self.scales).to(x.dtype)
        out = torch.matmul(x, w_dequant.t())
        if self.bias is not None:
            out += self.bias
        return out

def quantize_model_to_int8(model: nn.Module) -> nn.Module:
    """
    Recorre recursivamente todos los módulos del modelo y reemplaza
    todas las capas nn.Linear por capas Int8Linear.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, Int8Linear.from_float(child))
        else:
            quantize_model_to_int8(child)
    return model
