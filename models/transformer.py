import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight).type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute cos/sin tables for RoPE (real-valued, avoids complex ops)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, inv_freq).float()          # [end, dim/2]
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rotary_emb(q, k, cos_sin):
    """Apply RoPE using real-valued cos/sin tables.
    q,k: [B, n_heads, L, head_dim]; cos_sin: (cos, sin) each [end, dim/2]
    """
    cos, sin = cos_sin
    L = q.shape[-2]
    cos = cos[:L].view(1, 1, L, -1)
    sin = sin[:L].view(1, 1, L, -1)

    def rotate(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2)

    return rotate(q).type_as(q), rotate(k).type_as(k)


class GQA(nn.Module):
    """Grouped Query Attention with RoPE."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 dropout: float = 0.0, max_seq_len: int = 8192):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = dropout

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        rope_cos, rope_sin = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

    def repeat_kv(self, x, n_rep: int):
        if n_rep == 1:
            return x
        B, n, L, d = x.shape
        return x.unsqueeze(2).expand(B, n, n_rep, L, d).reshape(B, n * n_rep, L, d)

    def forward(self, x, mask=None, use_flash=True):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos_sin = (self.rope_cos[:L], self.rope_sin[:L])
        q, k = apply_rotary_emb(q, k, cos_sin)

        k = self.repeat_kv(k, self.n_rep)
        v = self.repeat_kv(v, self.n_rep)

        if use_flash and hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores += mask
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = int(hidden_mult * d_model * 2 / 3)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 dropout: float = 0.0, hidden_mult: int = 4, max_seq_len: int = 8192):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv_heads, dropout, max_seq_len)
        self.ffn = SwiGLU(d_model, hidden_mult, dropout)

    def forward(self, x, mask=None, use_flash=True):
        x = x + self.attn(self.norm1(x), mask, use_flash)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int,
                 n_kv_heads: Optional[int] = None, dropout: float = 0.0,
                 hidden_mult: int = 4, max_seq_len: int = 8192):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, n_kv_heads,
                                    dropout, hidden_mult, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
1
    def forward(self, x, mask=None, use_flash=True):
        for layer in self.layers:
            if self.training and x.requires_grad:
                x = checkpoint(layer, x, mask, use_flash, use_reentrant=False)
            else:
                x = layer(x, mask, use_flash)
        return self.norm(x)
