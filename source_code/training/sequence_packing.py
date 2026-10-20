# Copyright 2025 The Qwen Team and Alibaba Group.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Module: Sample Packing & Variable-Length Sequence Collator
# Eliminates wasted padding tokens by concatenating multiple variable-length
# training dialogues/prompts into single continuous tensors with block-diagonal causal masks.

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import torch


@dataclass
class PackedSequenceBatch:
    """
    Contenedor de secuencias empaquetadas sin relleno de padding.
    """
    input_ids: torch.Tensor           # (1, total_tokens)
    labels: torch.Tensor              # (1, total_tokens)
    cu_seqlens: torch.Tensor          # (num_samples + 1,) enteros acumulados [0, l1, l1+l2, ...]
    attention_mask_2d: torch.Tensor   # (1, 1, total_tokens, total_tokens) máscara causal en bloque
    max_seqlen: int
    num_samples: int


class SequencePacker:
    """
    Empaquetador de secuencias de alta eficiencia para entrenamiento con FlashAttention / GQA.
    Aplica Bin Packing voraz (First-Fit Decreasing) para maximizar el uso de VRAM y el throughput de tokens/segundo.
    """
    def __init__(self, max_tokens_per_pack: int = 4096, pad_token_id: int = 151643):
        self.max_tokens_per_pack = max_tokens_per_pack
        self.pad_token_id = pad_token_id

    @staticmethod
    def build_block_causal_mask(cu_seqlens: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        """
        Construye una máscara causal 2D diagonal por bloques.
        Garantiza que el token j de la secuencia k solo pueda atender a tokens i <= j
        pertenecientes a la misma secuencia k (sin contaminación entre ejemplos empaquetados).
        """
        total_len = cu_seqlens[-1].item()
        mask = torch.full((total_len, total_len), float("-inf"), device=device)

        # Construir máscara causal para cada bloque independiente
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start
            
            # Sub-matriz causal triangular inferior (0.0 en attendable, -inf en no attendable)
            sub_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
            mask[start:end, start:end] = sub_mask

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, total_len, total_len)

    def pack(
        self,
        samples: List[Dict[str, torch.Tensor]],
        device: Optional[torch.device] = None
    ) -> List[PackedSequenceBatch]:
        """
        Empaqueta una lista de diccionarios {'input_ids': LongTensor, 'labels': LongTensor}
        dentro de uno o más contenedores PackedSequenceBatch.
        """
        if not samples:
            return []

        # Ordenar por longitud decreciente para optimizar bin-packing (First-Fit Decreasing)
        sorted_samples = sorted(
            samples,
            key=lambda x: x["input_ids"].shape[-1],
            reverse=True
        )

        bins: List[List[Dict[str, torch.Tensor]]] = []
        bin_lengths: List[int] = []

        for sample in sorted_samples:
            s_len = sample["input_ids"].shape[-1]
            if s_len > self.max_tokens_per_pack:
                raise ValueError(
                    f"Sample length ({s_len}) exceeds max_tokens_per_pack ({self.max_tokens_per_pack})"
                )

            # Intentar ubicar en el primer bin con capacidad suficiente
            placed = False
            for b_idx, b_len in enumerate(bin_lengths):
                if b_len + s_len <= self.max_tokens_per_pack:
                    bins[b_idx].append(sample)
                    bin_lengths[b_idx] += s_len
                    placed = True
                    break

            if not placed:
                bins.append([sample])
                bin_lengths.append(s_len)

        packed_batches: List[PackedSequenceBatch] = []

        for b_samples in bins:
            all_input_ids = []
            all_labels = []
            cu_seqlens = [0]
            max_len = 0

            for s in b_samples:
                ids = s["input_ids"].view(-1)
                lbl = s.get("labels", ids).view(-1)
                all_input_ids.append(ids)
                all_labels.append(lbl)
                s_len = len(ids)
                cu_seqlens.append(cu_seqlens[-1] + s_len)
                if s_len > max_len:
                    max_len = s_len

            cat_input_ids = torch.cat(all_input_ids, dim=0).unsqueeze(0)  # (1, total_len)
            cat_labels = torch.cat(all_labels, dim=0).unsqueeze(0)        # (1, total_len)
            cu_tensor = torch.tensor(cu_seqlens, dtype=torch.int32)

            if device is not None:
                cat_input_ids = cat_input_ids.to(device)
                cat_labels = cat_labels.to(device)
                cu_tensor = cu_tensor.to(device)

            mask_2d = self.build_block_causal_mask(cu_tensor, device=cat_input_ids.device)

            packed_batches.append(
                PackedSequenceBatch(
                    input_ids=cat_input_ids,
                    labels=cat_labels,
                    cu_seqlens=cu_tensor,
                    attention_mask_2d=mask_2d,
                    max_seqlen=max_len,
                    num_samples=len(b_samples)
                )
            )

        return packed_batches

    @staticmethod
    def unpack_hidden_states(
        packed_tensor: torch.Tensor,
        cu_seqlens: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Desempaqueta un tensor 2D/3D (1, total_tokens, D) en una lista de tensores individuales (1, seq_len_i, D).
        """
        unpacked = []
        if packed_tensor.dim() == 3:
            packed_tensor = packed_tensor.squeeze(0)  # (total_tokens, D)

        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            unpacked.append(packed_tensor[start:end].unsqueeze(0))
        return unpacked
