# Copyright 2025 The Qwen Team and Alibaba Group.
# Apache License, Version 2.0 (the "License");
#
# Module: Sliding Window Attention (SWA) with Attention Sink Preservation
# Implements bounded O(W) attention footprint with initial sink preservation
# for infinite sequence autoregressive stability.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .qk_norm import QKNorm
from .gqa import repeat_kv

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention (SWA) with Attention Sinks.
    
    Tokens attend strictly to:
    1. Initial 'sink_tokens' positions (0 to S-1) to avoid attention collapse.
    2. A trailing window of size 'window_size' (i - W + 1 to i).
    
    Reduces memory complexity from O(T^2) to O(T * W).
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        window_size: int = 512,
        sink_tokens: int = 4,
        use_qk_norm: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.sink_tokens = sink_tokens
        self.scale = 1.0 / math.sqrt(head_dim)
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.qk_norm = QKNorm(head_dim) if use_qk_norm else None

    def construct_sliding_sink_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Creates a causal attention mask of shape (1, 1, seq_len, seq_len) where
        mask[q, k] = 0.0 if k < sink_tokens OR (q - k < window_size and k <= q),
        else -1e9.
        """
        q_idx = torch.arange(seq_len, device=device).unsqueeze(1) # (seq_len, 1)
        k_idx = torch.arange(seq_len, device=device).unsqueeze(0) # (1, seq_len)
        
        causal_valid = (k_idx <= q_idx)
        is_sink = (k_idx < self.sink_tokens)
        is_window = ((q_idx - k_idx) < self.window_size)
        
        allowed = causal_valid & (is_sink | is_window)
        mask = torch.full((seq_len, seq_len), -1e9, device=device, dtype=dtype)
        mask[allowed] = 0.0
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[nn.Module] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        if self.qk_norm is not None:
            q = self.qk_norm.q_norm(q)
            k = self.qk_norm.k_norm(k)
            
        q = q.transpose(1, 2) # (B, H, T, D)
        k = k.transpose(1, 2) # (B, H_kv, T, D)
        v = v.transpose(1, 2)
        
        if rotary_emb is not None:
            q, k = rotary_emb(q, k)
            
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        current_kv = (k, v)
        total_kv_len = k.shape[2]
        
        # Expand KV heads for GQA
        k_expanded = repeat_kv(k, self.num_heads // self.num_kv_heads)
        v_expanded = repeat_kv(v, self.num_heads // self.num_kv_heads)
        
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scale
        
        if seq_len > 1:
            # Prefill mask
            mask = self.construct_sliding_sink_mask(total_kv_len, q.device, q.dtype)
            if mask.shape[-2] != seq_len:
                mask = mask[:, :, -seq_len:, :]
            scores = scores + mask
        else:
            # Autoregressive single-token decode
            # query position is total_kv_len - 1
            curr_pos = total_kv_len - 1
            k_indices = torch.arange(total_kv_len, device=q.device)
            is_sink = k_indices < self.sink_tokens
            is_window = (curr_pos - k_indices) < self.window_size
            allowed = is_sink | is_window
            
            mask = torch.full((1, 1, 1, total_kv_len), -1e9, device=q.device, dtype=q.dtype)
            mask[:, :, :, allowed] = 0.0
            scores = scores + mask
            
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v_expanded) # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out), current_kv
