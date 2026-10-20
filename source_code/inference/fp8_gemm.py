# Copyright 2025 The Qwen Team and Alibaba Group.
# Apache License, Version 2.0 (the "License");
#
# Module: FP8 Dynamic Quantization and Tensor GEMM Emulation
# Implements hardware-aligned FP8 (E4M3FN and E5M2) execution with per-token
# and per-channel scaling factors for modern Tensor Core acceleration.

import torch
import torch.nn as nn
from typing import Tuple, Optional, Literal

FP8_MAX_E4M3 = 448.0
FP8_MAX_E5M2 = 57344.0

class FP8Dequantizer:
    """Utilities for Float8 quantization, scale extraction, and dequantization."""
    
    @staticmethod
    def quantize(
        tensor: torch.Tensor,
        format_type: Literal["e4m3", "e5m2"] = "e4m3",
        granularity: Literal["per_tensor", "per_channel"] = "per_channel"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes a floating point tensor to FP8 with dynamic scaling.
        Returns:
            (quantized_tensor, scale) where original ~ quantized.float() * scale
        """
        target_dtype = torch.float8_e4m3fn if format_type == "e4m3" else torch.float8_e5m2
        max_val = FP8_MAX_E4M3 if format_type == "e4m3" else FP8_MAX_E5M2
        
        eps = 1e-12
        if granularity == "per_tensor":
            amax = tensor.abs().max()
            scale = (amax / max_val).clamp(min=eps)
            scaled = (tensor / scale).clamp(-max_val, max_val)
        else: # per_channel along last dimension
            amax = tensor.abs().amax(dim=-1, keepdim=True)
            scale = (amax / max_val).clamp(min=eps)
            scaled = (tensor / scale).clamp(-max_val, max_val)
            
        quantized = scaled.to(target_dtype)
        return quantized, scale

    @staticmethod
    def dequantize(
        quantized: torch.Tensor,
        scale: torch.Tensor,
        target_dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Dequantizes FP8 tensor back to high-precision float using scale."""
        return quantized.to(target_dtype) * scale


class FP8Linear(nn.Module):
    """
    Drop-in Linear layer executing with FP8-quantized weights (E4M3 or E5M2)
    and per-token dynamic activation scaling.
    
    Saves 50% memory over FP16/BF16 and mimics Blackwell/Hopper Tensor Core pipeline.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        format_type: Literal["e4m3", "e5m2"] = "e4m3"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.format_type = format_type
        
        # Buffer for FP8 quantized weights and channel scales
        self.register_buffer(
            "weight_fp8",
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn if format_type == "e4m3" else torch.float8_e5m2)
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, 1, dtype=torch.float32)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, format_type: Literal["e4m3", "e5m2"] = "e4m3") -> "FP8Linear":
        """Factory method to construct FP8Linear by quantizing an existing nn.Linear."""
        fp8_layer = cls(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            bias=linear_layer.bias is not None,
            format_type=format_type
        )
        
        with torch.no_grad():
            w_fp8, w_scale = FP8Dequantizer.quantize(
                linear_layer.weight.data,
                format_type=format_type,
                granularity="per_channel"
            )
            fp8_layer.weight_fp8.copy_(w_fp8)
            fp8_layer.weight_scale.copy_(w_scale)
            
            if linear_layer.bias is not None:
                fp8_layer.bias.copy_(linear_layer.bias.data)
                
        return fp8_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes FP8 linear projection:
        1. Dynamically quantizes activations x to FP8 per-token.
        2. Performs scaled dequantization and matmul.
        """
        orig_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        
        # Dynamic per-token activation quantization
        x_fp8, x_scale = FP8Dequantizer.quantize(
            x_flat,
            format_type=self.format_type,
            granularity="per_channel"
        )
        
        # Dequantize weights and activations to computation precision
        # (On native hardware FP8 tensor cores this happens in SRAM registers)
        w_dequant = FP8Dequantizer.dequantize(self.weight_fp8, self.weight_scale, target_dtype=x.dtype)
        x_dequant = FP8Dequantizer.dequantize(x_fp8, x_scale, target_dtype=x.dtype)
        
        out = torch.matmul(x_dequant, w_dequant.t())
        if self.bias is not None:
            out = out + self.bias
            
        return out.view(*orig_shape[:-1], self.out_features)


def convert_linear_to_fp8(module: nn.Module, format_type: Literal["e4m3", "e5m2"] = "e4m3") -> nn.Module:
    """Recursively replaces all nn.Linear instances with FP8Linear in a module."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, FP8Linear.from_linear(child, format_type=format_type))
        else:
            convert_linear_to_fp8(child, format_type=format_type)
    return module
