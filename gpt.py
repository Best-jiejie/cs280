import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # q, k: [B, n_head, T, head_dim]
    cos = cos.squeeze(1).squeeze(0)  # [T, head_dim]
    sin = sin.squeeze(1).squeeze(0)  # [T, head_dim]
    cos = cos[position_ids].unsqueeze(1)  # [B, 1, T, head_dim]
    sin = sin[position_ids].unsqueeze(1)  # [B, 1, T, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        assert config.n_embd % self.n_head == 0
        self.head_dim = config.n_embd // self.n_head
        self.n_embd = config.n_embd
        self.use_flash_attention = config.use_flash_attention

        self.wq = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=config.block_size)

        if not self.use_flash_attention:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                         .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, position_ids, use_cache=False, past_key_value=None, return_attention=False):
        B, T, C = x.size()

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v, seq_len=self.rotary_emb.max_seq_len_cached)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        present_key_value = (k, v) if use_cache else None

        # Repeat KV for GQA
        if self.n_kv_head < self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        att_weights = None

        if self.use_flash_attention and not return_attention:
            # Flash Attention natively supports causal masking
            y = F.scaled_dot_product_attention(q, k, v, is_causal=(past_key_value is None))
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if past_key_value is None:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att_weights = att
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        if return_attention:
            return y, present_key_value, att_weights
        return y, present_key_value

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, position_ids, use_cache=False, past_key_value=None, return_attention=False):
        if return_attention:
            attn_out, present_key_value, att_weights = self.attn(self.ln_1(x), position_ids, use_cache, past_key_value, return_attention)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, present_key_value, att_weights
        else:
            attn_out, present_key_value = self.attn(self.ln_1(x), position_ids, use_cache, past_key_value)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, present_key_value

class GPTConfig:
    def __init__(self, vocab_size=50257, block_size=256, n_layer=6, n_head=6, n_embd=384, n_kv_head=None, use_flash_attention=True):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.use_flash_attention = use_flash_attention

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, use_cache=False, past_key_values=None, return_attention=False, pad_token_id=-1):
        device = idx.device
        b, t = idx.size()

        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2)
            position_ids = torch.arange(past_length, past_length + t, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)
        else:
            position_ids = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)

        x = self.transformer.wte(idx)

        present_key_values = [] if use_cache else None
        all_att_weights = [] if return_attention else None

        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            if return_attention:
                x, present_kv, att_weights = block(x, position_ids, use_cache=use_cache, past_key_value=past_kv, return_attention=True)
                all_att_weights.append(att_weights)
            else:
                x, present_kv = block(x, position_ids, use_cache=use_cache, past_key_value=past_kv)
            
            if use_cache:
                present_key_values.append(present_kv)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token_id)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if return_attention:
            return logits, loss, present_key_values, all_att_weights
        return logits, loss, present_key_values