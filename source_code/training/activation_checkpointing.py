# Copyright 2025 The Qwen Team and Alibaba Group.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Module: Selective Activation Checkpointing Engine
# Reduces VRAM consumption during backpropagation by discarding intermediate
# activation tensors in transformer blocks and recomputing them on-demand.

from typing import List, Optional, Type, Any
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointedBlockWrapper(nn.Module):
    """
    Envoltura para bloques decodificadores que activa gradient checkpointing
    mediante el moderno backend no reentrante de PyTorch.
    """
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        # Si no requiere gradiente o no estamos entrenando, forward normal sin sobrecoste
        has_grad = any(isinstance(a, torch.Tensor) and a.requires_grad for a in args) or any(
            isinstance(v, torch.Tensor) and v.requires_grad for v in kwargs.values()
        )
        if not has_grad or not self.training:
            return self.block(*args, **kwargs)

        def custom_forward(*f_args):
            return self.block(*f_args, **kwargs)

        # Usar use_reentrant=False para compatibilidad con autograd avanzado y DDP
        return checkpoint(custom_forward, *args, use_reentrant=False)


def apply_activation_checkpointing(
    model: nn.Module,
    every_n_layers: int = 1,
    target_block_types: Optional[List[Type[nn.Module]]] = None
) -> int:
    """
    Aplica checkpointing selectivo a las capas del modelo.
    
    Args:
        model: Instancia de modelo (StandaloneQwen3ForCausalLM o nn.Module).
        every_n_layers: Intervalo de capas a checkpointing (1 = todas las capas, 2 = alternadas).
        target_block_types: Tipos de clase a interceptar (por defecto busca atributos .layers o .blocks).
    
    Returns:
        Número de bloques configurados con checkpointing.
    """
    checkpointed_count = 0
    layers = None

    # Intentar detectar la lista de capas decodificadoras
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "blocks"):
        layers = model.blocks

    if layers is None or not isinstance(layers, (nn.ModuleList, list)):
        # Búsqueda recursiva si no se encuentra la lista estándar
        for name, module in model.named_children():
            if "layer" in name.lower() or "block" in name.lower():
                if isinstance(module, nn.ModuleList):
                    layers = module
                    break

    if layers is None:
        return 0

    for idx in range(len(layers)):
        if (idx % every_n_layers) == 0:
            original_layer = layers[idx]
            if not isinstance(original_layer, CheckpointedBlockWrapper):
                layers[idx] = CheckpointedBlockWrapper(original_layer)
                checkpointed_count += 1

    return checkpointed_count
