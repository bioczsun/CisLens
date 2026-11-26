"""CisLens network; full-input ChromEPI parameter names are preserved."""

import torch
import torch.nn.functional as F
from torch import nn
import math
from einops.layers.torch import Rearrange


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}


def ConvBlock(dim, dim_out=None, kernel_size=1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        nn.GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding=kernel_size // 2),
    )


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-08, dim: int = 1536):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x) * self.weight.to(dtype=x.dtype)
        return output


def build_rope_cache(
    seq_len: int, head_dim: int, device, base: float = 10000.0, dtype=torch.float32
):
    """
    returns:
      cos, sin: [seq_len, head_dim], usually reshaped for broadcasting
    """
    assert head_dim % 2 == 0
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    cos = torch.repeat_interleave(freqs.cos(), repeats=2, dim=-1)
    sin = torch.repeat_interleave(freqs.sin(), repeats=2, dim=-1)
    return (cos, sin)


def rotate_half(x: torch.Tensor):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return x_rot


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    q,k: [B, T, H, D]
    cos, sin: [T, D] or already reshaped to [1, T, 1, D]
    """
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return (q, k)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.base = base
        self.register_buffer("cos_cache", None, persistent=False)
        self.register_buffer("sin_cache", None, persistent=False)

    @torch.no_grad()
    def _update_cache(self, seq_len: int, device, dtype):
        cos, sin = build_rope_cache(
            seq_len, self.head_dim, device=device, base=self.base, dtype=torch.float32
        )
        self.cos_cache = cos
        self.sin_cache = sin

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        q,k: [B, T, H, D]
        """
        B, T, H, D = q.shape
        if (
            self.cos_cache is None
            or self.cos_cache.shape[0] < T
            or self.cos_cache.device != q.device
        ):
            self._update_cache(T, device=q.device, dtype=q.dtype)
        cos = self.cos_cache[:T].to(dtype=q.dtype)
        sin = self.sin_cache[:T].to(dtype=q.dtype)
        return apply_rope(q, k, cos, sin)


class MHABlock(nn.Module):
    def __init__(
        self,
        dim: int = 1536,
        num_heads: int = 8,
        qk_dim: int = 64,
        v_dim: int = 64,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        eps: float = 1e-06,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len
        self.pre_norm = RMSNorm(dim=dim, eps=eps)
        self.pos_norm = RMSNorm(dim=dim, eps=eps)
        self.q_proj = nn.Linear(dim, num_heads * qk_dim, bias=False)
        self.k_proj = nn.Linear(dim, 1 * qk_dim, bias=False)
        self.v_proj = nn.Linear(dim, 1 * v_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * v_dim, dim, bias=True)
        self.q_ln = nn.LayerNorm(qk_dim, eps=eps)
        self.k_ln = nn.LayerNorm(qk_dim, eps=eps)
        self.v_ln = nn.LayerNorm(v_dim, eps=eps)
        self.rope = RoPE(head_dim=qk_dim, base=10000.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_bias: torch.Tensor):
        b, s, c = x.shape
        assert c == self.dim, f"Expected C={self.dim}, got {c}"
        if attention_bias is None:
            attention_bias = torch.zeros((b, self.num_heads, s, s), device=x.device, dtype=x.dtype)
        else:
            assert attention_bias.shape == (
                b,
                self.num_heads,
                s,
                s,
            ), f"Expected attention_bias [B,{self.num_heads},S,S], got {attention_bias.shape}"
        x = self.pre_norm(x)
        q = self.q_proj(x).view(b, s, self.num_heads, self.qk_dim)
        q = self.q_ln(q)
        k = self.k_proj(x).view(b, s, 1, self.qk_dim)
        k = self.k_ln(k)
        k = k.expand(b, s, self.num_heads, self.qk_dim)
        v = self.v_proj(x).view(b, s, 1, self.v_dim)
        v = self.v_ln(v)
        v = v.expand(b, s, self.num_heads, self.v_dim)
        q, k = self.rope(q, k)
        scale = 1.0 / math.sqrt(self.qk_dim)
        attn_logits = torch.einsum("bshd,bthd->bhst", q, k) * scale
        attn_logits = torch.tanh((attn_logits + attention_bias) / 5.0) * 5.0
        attn_weights = F.softmax(attn_logits, dim=-1)
        y = torch.einsum("bhst,bthd->bshd", attn_weights, v)
        y = y.reshape(b, s, self.num_heads * self.v_dim)
        y = self.o_proj(y)
        y = self.pos_norm(y)
        y = self.dropout(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.pre_norm = RMSNorm(dim=dim)
        self.pos_norm = RMSNorm(dim=dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.pos_norm(x)
        x = self.dropout(x)
        return x


class AttentionBiasBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, repeat_factor: int = 16, eps: float = 1e-06):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.repeat_factor = repeat_factor
        self.norm = RMSNorm(dim=dim)
        self.proj = nn.Linear(dim, num_heads, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape [B,P,P,F], got {tuple(x.shape)}")
        b, p1, p2, f = x.shape
        if p1 != p2:
            raise ValueError(f"Expected square [P,P], got P1={p1}, P2={p2}")
        x = F.gelu(self.norm(x))
        x = self.proj(x)
        rf = self.repeat_factor
        x = x.repeat_interleave(rf, dim=1).repeat_interleave(rf, dim=2)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 1536,
        num_heads: int = 8,
        qk_dim: int = 64,
        v_dim: int = 64,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        eps: float = 1e-06,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mha = MHABlock(
            dim=dim,
            num_heads=num_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            eps=eps,
        )
        self.ffn = FeedForward(dim=dim)

    def forward(self, x: torch.Tensor, attention_bias: torch.Tensor):
        x = self.mha(x, attention_bias=attention_bias) + x
        x = self.ffn(x) + x
        return x


def central_mask_features(
    sequence_length: int, feature_size: int, device=None, dtype=None
) -> torch.Tensor:
    L = sequence_length
    Fsz = feature_size
    assert Fsz % 2 == 0, "feature_size must be even"
    device = device if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.float32
    rel = torch.arange(2 * L - 1, device=device, dtype=dtype) - (L - 1)
    n = Fsz // 2
    start = torch.tensor(1.0, device=device, dtype=dtype)
    end = torch.tensor(float(L - n + 1), device=device, dtype=dtype)
    logs = torch.linspace(start.log(), end.log(), steps=n + 1, device=device, dtype=dtype)[:-1]
    geom = logs.exp()
    center_widths = torch.arange(n, device=device, dtype=dtype) + geom
    emb = (center_widths[None, :] > rel.abs()[:, None]).to(dtype)
    signed = torch.sign(rel)[:, None] * emb
    return torch.cat([emb, signed], dim=-1)


def relative_shift(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, P, 2P-1, H]
    return: [B, P, P, H]
    """
    if x.dim() != 4:
        raise ValueError(f"relative_shift expects [B,P,2P-1,H], got {tuple(x.shape)}")
    B, P, D, H = x.shape
    if D != 2 * P - 1:
        raise ValueError(f"Expected D=2*P-1={2 * P - 1}, got {D}")
    x = F.pad(x, (0, 0, 1, 0))
    x = x.reshape(B, D + 1, P, H)
    x = x[:, 1:, :, :]
    x = x.reshape(B, P, D, H)
    return x[:, :, :P, :]


class SequenceToPairBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        pool_size: int = 16,
        num_heads: int = 32,
        head_dim: int = 64,
        pos_feature_size: int = 64,
        out_pair_dim: int = 128,
        dropout_p: float = 0.1,
        eps: float = 1e-06,
    ):
        super().__init__()
        self.dim = dim
        self.pool_size = pool_size
        self.H = num_heads
        self.D = head_dim
        self.pos_feature_size = pos_feature_size
        self.out_pair_dim = out_pair_dim
        self.norm = RMSNorm(dim=dim, eps=eps)
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.pos_proj = nn.Linear(pos_feature_size, num_heads * head_dim, bias=True)
        self.q_r_bias = nn.Parameter(torch.zeros(1, 1, num_heads, head_dim))
        self.k_r_bias = nn.Parameter(torch.zeros(1, 1, num_heads, head_dim))
        self.y_q_proj = nn.Linear(dim, out_pair_dim, bias=False)
        self.y_k_proj = nn.Linear(dim, out_pair_dim, bias=False)
        self.a_proj = nn.Linear(num_heads, out_pair_dim, bias=True)
        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer("_cached_pos_features", torch.empty(0), persistent=False)
        self._cached_P = None

    def _downsample_avgpool(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        if S % self.pool_size != 0:
            raise ValueError(f"S={S} must be divisible by pool_size={self.pool_size}")
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)
        return x.transpose(1, 2)

    def _get_pos_features(self, P: int, device, dtype) -> torch.Tensor:
        if (
            self._cached_P != P
            or self._cached_pos_features.numel() == 0
            or self._cached_pos_features.device != device
            or (self._cached_pos_features.dtype != dtype)
        ):
            pf = central_mask_features(P, self.pos_feature_size, device=device, dtype=dtype)
            self._cached_pos_features = pf
            self._cached_P = P
        return self._cached_pos_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        if C != self.dim:
            raise ValueError(f"Expected C={self.dim}, got {C}")
        x = self._downsample_avgpool(x)
        x = self.norm(x)
        B, P, _ = x.shape
        q = self.q_proj(x).view(B, P, self.H, self.D)
        k = self.k_proj(x).view(B, P, self.H, self.D)
        pos_features = self._get_pos_features(P, x.device, x.dtype)
        pos_encoding = self.pos_proj(pos_features).view(2 * P - 1, self.H, self.D)
        rel_q_raw = torch.einsum("bphd,rhd->bprh", q + self.q_r_bias, pos_encoding)
        rel_k_raw = torch.einsum("bphd,rhd->bprh", k + self.k_r_bias, pos_encoding)
        rel_q_a = relative_shift(rel_q_raw)
        rel_k_a = relative_shift(rel_k_raw)
        a_content = torch.einsum("bqhd,bkhd->bqkh", q, k)
        a = a_content + 0.5 * (rel_q_a + rel_k_a.transpose(1, 2))
        gx = F.gelu(x)
        y_q = self.y_q_proj(gx)
        y_k = self.y_k_proj(gx)
        a_feat = self.a_proj(a)
        pair_activations = a_feat + y_q[:, :, None, :] + y_k[:, None, :, :]
        return self.dropout(pair_activations)


class RowAttentionBlock(nn.Module):
    """
    Implements:

      x = RMSNorm(pair_input)
      k = Linear(128, bias=False)(x)
      q = Linear(128, bias=False)(x)
      v = Linear(128)(x)
      att = einsum('bpPf,bpkf->bpPk', q, k) / sqrt(128)
      out = einsum('bpPk,bpkf->bpPf', softmax(att, axis=3), v)
      return Dropout(out)

    Shape convention:
      pair_input: [B, P, P, F]
      Here we attend along the *columns* (k index) for each row p (and batch B).
      Output: [B, P, P, 128] (or [B,P,P,F] if you add an output proj; pseudocode doesn't)
    """

    def __init__(
        self, in_dim: int, proj_dim: int = 128, dropout_p: float = 0.1, eps: float = 1e-06
    ):
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.norm = RMSNorm(dim=in_dim, eps=eps)
        self.k_proj = nn.Linear(in_dim, proj_dim, bias=False)
        self.q_proj = nn.Linear(in_dim, proj_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, proj_dim, bias=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, pair_input: torch.Tensor) -> torch.Tensor:
        """
        pair_input: [B, P, P, F]
        returns: [B, P, P, proj_dim]
        """
        if pair_input.dim() != 4:
            raise ValueError(f"Expected [B,P,P,F], got {tuple(pair_input.shape)}")
        x = self.norm(pair_input)
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        att = torch.einsum("bpqf,bpkf->bpqk", q, k) / math.sqrt(self.proj_dim)
        w = F.softmax(att, dim=3)
        out = torch.einsum("bpqk,bpkf->bpqf", w, v)
        return self.dropout(out)


class PairMLPBlock(nn.Module):
    """
    Implements:

      x = RMSNorm(pair_input)
      x = Linear(2F)(x)                # F -> 2F
      x = Linear(F)(ReLU(x))           # 2F -> F
      return Dropout(x)

    Input/Output: [B,P,P,F]
    """

    def __init__(self, dim: int, dropout_p: float = 0.1, eps: float = 1e-06):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim=dim, eps=eps)
        self.fc1 = nn.Linear(dim, 2 * dim, bias=True)
        self.fc2 = nn.Linear(2 * dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, pair_input: torch.Tensor) -> torch.Tensor:
        if pair_input.dim() != 4 or pair_input.size(-1) != self.dim:
            raise ValueError(f"Expected [B,P,P,{self.dim}], got {tuple(pair_input.shape)}")
        x = self.norm(pair_input)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return self.dropout(x)


class PairUpdateBlock(nn.Module):
    def __init__(
        self,
        sequence_to_pair_block: nn.Module,
        row_attn_block: nn.Module,
        pair_mlp_block: nn.Module,
    ):
        super().__init__()
        self.sequence_to_pair_block = sequence_to_pair_block
        self.row_attn_block = row_attn_block
        self.pair_mlp_block = pair_mlp_block

    def forward(
        self, sequence_input: torch.Tensor, pair_input: torch.Tensor | None = None
    ) -> torch.Tensor:
        y = self.sequence_to_pair_block(sequence_input)
        x = y if pair_input is None else pair_input + y
        x = x + self.row_attn_block(x)
        x = x + self.pair_mlp_block(x)
        return x


class CisLens(nn.Module):
    def __init__(
        self,
        dim=960,
        qk_dim=64,
        v_dim=64,
        num_head=8,
        max_seq_len=512,
        depth=6,
        pool_size=4,
        output_heads=None,
        mid_dim=128,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.depth = depth
        self.output_heads = output_heads if output_heads is not None else {"epi": 15}
        self.pool_size = pool_size
        epi_norm = nn.BatchNorm1d(8)
        self.res_blocks_epi = nn.Sequential(
            nn.AvgPool1d(32), nn.Conv1d(1, 8, 3, padding=1), epi_norm, nn.GELU(), nn.AvgPool1d(4)
        )
        self.conv_end = nn.Conv1d(in_channels=self.dim + 8, out_channels=self.dim, kernel_size=1)
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim, num_heads=num_head, qk_dim=qk_dim, v_dim=v_dim, max_seq_len=max_seq_len
                )
                for _ in range(self.depth)
            ]
        )
        self.final_pointwise = nn.Sequential(
            Rearrange("b n d -> b d n"),
            ConvBlock(self.dim, self.dim * 2, 1),
            Rearrange("b d n -> b n d"),
            nn.Dropout(0.1),
            nn.GELU(),
        )
        self.add_heads(**self.output_heads)
        attentiion_bias_block = AttentionBiasBlock(
            dim=mid_dim, num_heads=num_head, repeat_factor=pool_size
        )
        sequence_to_pair_block = SequenceToPairBlock(
            dim=self.dim,
            pool_size=self.pool_size,
            num_heads=32,
            head_dim=64,
            pos_feature_size=64,
            out_pair_dim=mid_dim,
        )
        row_attn_block = RowAttentionBlock(in_dim=mid_dim, proj_dim=mid_dim)
        pair_mlp_block = PairMLPBlock(dim=mid_dim)
        pair_update_block = PairUpdateBlock(sequence_to_pair_block, row_attn_block, pair_mlp_block)
        self.pair_update_block = pair_update_block
        self.attentiion_bias_block = attentiion_bias_block
        self.hicnorm = RMSNorm(dim=mid_dim, eps=1e-06)
        self.hicdecoder = nn.Sequential(nn.GELU(), nn.Conv2d(mid_dim, 1, kernel_size=1))

    def add_heads(self, **kwargs):
        self._heads = nn.ModuleDict(
            map_values(
                lambda features: nn.Sequential(nn.Linear(self.dim * 2, features), nn.Softplus()),
                kwargs,
            )
        )

    @property
    def heads(self):
        return self._heads

    def forward(self, epi: torch.Tensor, embedding: torch.Tensor):
        epi_signal = self.res_blocks_epi(epi)
        epi_signal = F.adaptive_avg_pool1d(epi_signal, embedding.size(2))
        epi_out = torch.cat([epi_signal, embedding], dim=1)
        emb = self.conv_end(epi_out).transpose(1, 2)
        pair_emb = None
        for index, block in enumerate(self.transformer):
            if index % 2 == 0:
                pair_emb = self.pair_update_block(emb, pair_emb)
            emb = block(emb, attention_bias=self.attentiion_bias_block(pair_emb))
        epi_emb = self.final_pointwise(emb)
        out = map_values(lambda fn: fn(epi_emb), self._heads)
        pair_emb = 0.5 * (pair_emb + pair_emb.transpose(1, 2))
        pair_emb = self.hicnorm(pair_emb)
        hic_out = self.hicdecoder(pair_emb.permute(0, 3, 1, 2)).squeeze(1)
        return (out, hic_out)
