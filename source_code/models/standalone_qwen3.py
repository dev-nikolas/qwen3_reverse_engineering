import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

class StandaloneQwen3Config:
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        tie_word_embeddings: bool = True
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings

class StandaloneRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)

class StandaloneRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 32768, theta: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch, seq_len]
        freqs = position_ids.unsqueeze(-1).float() * self.inv_freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # cos, sin: [batch, 1, seq_len, head_dim]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

class StandaloneQwen3Attention(nn.Module):
    def __init__(self, config: StandaloneQwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK-Norm nativo de Qwen3
        self.q_norm = StandaloneRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = StandaloneRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # QK-Norm por cabeza
        q = self.q_norm(q).transpose(1, 2) # [batch, num_heads, seq_len, head_dim]
        k = self.k_norm(k).transpose(1, 2) # [batch, num_kv_heads, seq_len, head_dim]
        v = v.transpose(1, 2)

        # RoPE
        cos_u = cos.unsqueeze(1) # [batch, 1, seq_len, head_dim]
        sin_u = sin.unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos_u, sin_u)

        # Repetición de claves y valores para GQA
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # Producto punto escalado
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        context = torch.matmul(attn_weights, v) # [batch, num_heads, seq_len, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_proj(context)

class StandaloneQwen3MLP(nn.Module):
    def __init__(self, config: StandaloneQwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class StandaloneQwen3DecoderLayer(nn.Module):
    def __init__(self, config: StandaloneQwen3Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = StandaloneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = StandaloneQwen3Attention(config, layer_idx)
        self.post_attention_layernorm = StandaloneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = StandaloneQwen3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Pre-norm Self Attention con residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        # Pre-norm MLP con residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class StandaloneQwen3Model(nn.Module):
    def __init__(self, config: StandaloneQwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = StandaloneRotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
        self.layers = nn.ModuleList([StandaloneQwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = StandaloneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        cos, sin = self.rotary_emb(position_ids)

        # Máscara causal triangular
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, seq_len]

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, cos, sin, causal_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states

class StandaloneQwen3ForCausalLM(nn.Module):
    def __init__(self, config: StandaloneQwen3Config):
        super().__init__()
        self.config = config
        self.model = StandaloneQwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits

    @classmethod
    def from_safetensors(cls, model_dir: str, dtype: torch.dtype = torch.float32):
        """Carga los pesos binarios Safetensors directamente en el modelo puro."""
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        config = StandaloneQwen3Config(
            vocab_size=cfg_dict.get("vocab_size", 151936),
            hidden_size=cfg_dict.get("hidden_size", 2048),
            intermediate_size=cfg_dict.get("intermediate_size", 6144),
            num_hidden_layers=cfg_dict.get("num_hidden_layers", 28),
            num_attention_heads=cfg_dict.get("num_attention_heads", 16),
            num_key_value_heads=cfg_dict.get("num_key_value_heads", 8),
            head_dim=cfg_dict.get("head_dim", 128),
            max_position_embeddings=cfg_dict.get("max_position_embeddings", 32768),
            rms_norm_eps=cfg_dict.get("rms_norm_eps", 1e-6),
            rope_theta=cfg_dict.get("rope_theta", 1000000.0)
        )

        model = cls(config).to(dtype)
        state_dict = {}

        # Cargamos los shards safetensors
        shard1 = os.path.join(model_dir, "model-00001-of-00002.safetensors")
        shard2 = os.path.join(model_dir, "model-00002-of-00002.safetensors")

        print(f"[*] Cargando shard 1: {shard1}")
        s1 = load_file(shard1)
        state_dict.update(s1)

        if os.path.exists(shard2):
            print(f"[*] Cargando shard 2: {shard2}")
            s2 = load_file(shard2)
            state_dict.update(s2)

        # Si lm_head no está en state_dict pero está ligado a embed_tokens
        if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        # Convertimos tipos si es necesario
        for k in state_dict:
            state_dict[k] = state_dict[k].to(dtype)

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"[+] Modelo Standalone cargado exitosamente. Parámetros: {sum(p.numel() for p in model.parameters()):,}")
        return model
