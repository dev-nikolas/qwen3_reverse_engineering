# Copyright 2026 Alibaba Group and the Qwen Team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

"""
Motor de Entrenamiento y Ajuste Fino Supervisado (SFT & Continual Pre-Training)
==============================================================================
Orquesta el ciclo completo de entrenamiento y backpropagation de Qwen3-1.7B
usando exclusivamente componentes nativos de la arquitectura de ingeniería inversa.

Características:
  - Empaquetamiento voraz de secuencias (Sequence Packing sin relleno de padding).
  - Checkpointing selectivo de activaciones para optimizar el consumo de VRAM.
  - Optimizador AdamW acoplado con Scheduler de Decaimiento Coseno y Warmup.
  - Recorte de gradientes (Gradient Clipping a norma 1.0) para prevenir divergencia.
  - Cálculo de perplejidad y métricas de throughput (tokens/segundo).
"""

import os
import sys
import math
import time
from typing import List, Dict, Any, Optional
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from .sequence_packing import SequencePacker, PackedBatch
from .activation_checkpointing import apply_activation_checkpointing
from models.standalone_qwen3 import StandaloneQwen3ForCausalLM, StandaloneQwen3Config


class SFTTrainer:
    """
    Entrenador de alta eficiencia para Qwen3.
    """
    def __init__(
        self,
        model: StandaloneQwen3ForCausalLM,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 10,
        max_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        use_checkpointing: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        if use_checkpointing:
            apply_activation_checkpointing(self.model, every_n_layers=2)

        # Configurar AdamW excluyendo decay en biases y RMSNorm
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name.lower() or "bias" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )

        self.global_step = 0
        self.packer = SequencePacker(max_tokens_per_pack=2048)

    def get_lr(self, step: int) -> float:
        """Calcula el learning rate con Linear Warmup + Cosine Annealing."""
        if step < self.warmup_steps:
            return self.learning_rate * float(step + 1) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr = self.learning_rate * 0.1
        return min_lr + (self.learning_rate - min_lr) * cosine_decay

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Cálculo de entropía cruzada causal estándar (desplazando logits en 1).
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return loss

    def train_step(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Ejecuta un paso atómico de entrenamiento.
        """
        self.model.train()
        input_ids = input_ids.to(self.device)
        if labels is None:
            labels = input_ids.clone()
        labels = labels.to(self.device)

        # Actualizar Learning Rate
        current_lr = self.get_lr(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        start_t = time.time()
        logits = self.model(input_ids)
        loss = self.compute_loss(logits, labels)
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        grad_norm = 0.0
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm))
            self.optimizer.step()
            self.optimizer.zero_grad()

        elapsed = time.time() - start_t
        tokens_count = input_ids.numel()
        throughput = tokens_count / max(1e-5, elapsed)
        perplexity = math.exp(min(20.0, loss.item()))

        self.global_step += 1

        return {
            "step": self.global_step,
            "loss": loss.item(),
            "perplexity": perplexity,
            "lr": current_lr,
            "grad_norm": grad_norm,
            "tokens_per_sec": throughput
        }

    def train_epoch(self, dataset_samples: List[Dict[str, torch.Tensor]]) -> List[Dict[str, float]]:
        """
        Empaqueta secuencias vorazmente y entrena todas las muestras.
        """
        packed_batches = self.packer.pack(dataset_samples)
        metrics = []

        for batch in packed_batches:
            step_stats = self.train_step(batch.input_ids)
            metrics.append(step_stats)

        return metrics
