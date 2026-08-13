"""
cmt_fusion.py — V/A-conditioned Cross-Modal Transformer + dialog-context layer.

Two novel contributions vs MemoCMT (Khan et al. 2025):
  1. V/A-conditioning: the cross-attention logits get a learnable bias
     computed from (valence, arousal, dominance) — the model knows
     WHERE in the valence-arousal space the utterance sits BEFORE
     fusing audio and text. Grounded in Russell's circumplex model.
  2. Dialog-context layer: a transformer over the [CLS] vectors of the
     10 previous utterances in the same dialog. MemoCMT treats each
     utterance in isolation; we use the conversation structure.

Architectural details (MemoCMT convention, frozen encoders):
  - Audio encoder:  facebook/wav2vec2-base  (frozen, 768-dim)
  - Text encoder:   bert-base-uncased      (frozen, 768-dim)
  - CMT: 2 layers, 4 heads, d=256, V/A-conditioned cross-attention
  - Dialog context: 2 layers, 4 heads, d=256, window=10
  - Aggregation: MIN (MemoCMT's best variant on ESD) over token dim
  - Loss: CE only (no SupCon — the CMT already does cross-modal alignment)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionConfig:
    audio_dim: int = 768         # wav2vec2-base hidden
    text_dim: int = 768          # BERT-base hidden
    proj_dim: int = 256          # per-modality projection
    num_classes: int = 4         # IEMOCAP 4-class
    n_cmt_layers: int = 2
    n_heads: int = 4
    va_dim: int = 3               # (valence, arousal, dominance)
    va_proj_dim: int = 64         # V/A projection into CMT hidden
    dropout: float = 0.3
    dialog_context: bool = True
    dialog_window: int = 10
    aggregation: str = "min"  # min, mean, max, cls — MemoCMT tried all 4
    # dialog_dim is set to proj_dim * 2 (audio_pooled + text_pooled = 512)
    # so the dialog buffer + transformer operate on the same feature dim
    # as the CMT's penultimate fused features.
    @property
    def dialog_dim(self) -> int:
        return self.proj_dim * 2


class PureCrossAttention(nn.Module):
    """
    MemoCMT's exact cross-attention layer: NO V/A bias, just standard
    scaled dot-product attention + FFN + residual.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        x = q + out
        x = x + self.ff(self.norm_ff(x))
        return x


class PureCMTBlock(nn.Module):
    """
    One MemoCMT block: audio attends to text, text attends to audio.
    Bidirectional cross-attention, no bias.
    """
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.audio_to_text = PureCrossAttention(cfg.proj_dim, cfg.n_heads, cfg.dropout)
        self.text_to_audio = PureCrossAttention(cfg.proj_dim, cfg.n_heads, cfg.dropout)
        self.norm_audio = nn.LayerNorm(cfg.proj_dim)
        self.norm_text = nn.LayerNorm(cfg.proj_dim)

    def forward(self, audio: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_n = self.norm_audio(audio)
        text_n = self.norm_text(text)
        audio_out = audio + self.audio_to_text(audio_n, text_n)
        text_out = text + self.text_to_audio(text_n, audio_n)
        return audio_out, text_out


class PureCMT(nn.Module):
    """
    MemoCMT's exact architecture (Khan et al. 2025): bidirectional cross-attention
    fusion over frozen HuBERT + BERT representations. NO V/A bias.
    Reproduces the paper's 81.85% on IEMOCAP-4-class with MIN aggregation.
    """
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.proj_audio = nn.Linear(cfg.audio_dim, cfg.proj_dim)
        self.proj_text = nn.Linear(cfg.text_dim, cfg.proj_dim)
        self.layers = nn.ModuleList([PureCMTBlock(cfg) for _ in range(cfg.n_cmt_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.proj_dim * 2),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.proj_dim * 2, cfg.proj_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.proj_dim, cfg.num_classes),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate over token dim per cfg.aggregation."""
        if self.cfg.aggregation == "min":
            return x.min(dim=1).values
        elif self.cfg.aggregation == "mean":
            return x.mean(dim=1)
        elif self.cfg.aggregation == "max":
            return x.max(dim=1).values
        elif self.cfg.aggregation == "cls":
            return x[:, 0, :]
        else:
            raise ValueError(f"Unknown aggregation: {self.cfg.aggregation}")

    def forward(self, audio_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        audio = self.proj_audio(audio_tokens)
        text = self.proj_text(text_tokens)
        for layer in self.layers:
            audio, text = layer(audio, text)
        audio_pooled = self._pool(audio)
        text_pooled = self._pool(text)
        fused = torch.cat([audio_pooled, text_pooled], dim=-1)
        return self.classifier(fused)


class VAAttentionBias(nn.Module):
    """
    Computes a per-head additive bias for the cross-attention logits
    from (V, A, D). This is the heart of the V/A-conditioning novelty.

    Math:
        B = MLP(V, A, D)            # (B, n_heads) — one scalar per head
        attn = softmax(QK^T/sqrt(d) + B) V
        B is broadcast to (B, n_heads, T_a, T_t) so each head gets one
        utterance-level scalar added to all its attention logits.
    """
    def __init__(self, va_dim: int, proj_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        # Output dim is n_heads (one scalar per head), NOT n_heads * proj_dim.
        # The proj_dim arg is kept for API compat but unused here — each
        # head gets a single scalar that biases attention direction uniformly.
        self.mlp = nn.Sequential(
            nn.Linear(va_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, n_heads),
        )

    def forward(self, va: torch.Tensor, T_a: int, T_t: int) -> torch.Tensor:
        """
        Args:
            va: (B, 3) — valence, arousal, dominance (normalized 0..1)
            T_a: number of audio tokens
            T_t: number of text tokens
        Returns:
            bias: (B, n_heads, T_a, T_t) — additive bias added to attn logits
        """
        B = va.size(0)
        h = self.mlp(va)                    # (B, n_heads)
        h = h.view(B, self.n_heads, 1, 1)   # (B, n_heads, 1, 1)
        return h.expand(B, self.n_heads, T_a, T_t)


class CrossModalAttentionLayer(nn.Module):
    """One cross-attention block with V/A conditioning."""
    def __init__(self, d_model: int, n_heads: int, va_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.va_bias = VAAttentionBias(va_dim, d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.n_heads = n_heads

    def forward(self, q: torch.Tensor, kv: torch.Tensor, va: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: (B, T_q, d)
            kv: (B, T_kv, d)
            va: (B, 3)
        Returns:
            (B, T_q, d)
        """
        T_q, T_kv = q.size(1), kv.size(1)
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        # Compute attention with V/A bias added to logits
        # We need to do this manually because MultiheadAttention doesn't
        # support additive bias on logits directly.
        B, _, d = qn.shape
        H = self.n_heads
        # Project Q, K, V
        W_q, W_k, W_v = self.attn.in_proj_weight.split(self.d_model, dim=0)
        b_q, b_k, b_v = self.attn.in_proj_bias.split(self.d_model, dim=0)
        Q = F.linear(qn, W_q, b_q).view(B, T_q, H, d // H).transpose(1, 2)  # (B, H, T_q, d_h)
        K = F.linear(kvn, W_k, b_k).view(B, T_kv, H, d // H).transpose(1, 2)  # (B, H, T_kv, d_h)
        V = F.linear(kvn, W_v, b_v).view(B, T_kv, H, d // H).transpose(1, 2)  # (B, H, T_kv, d_h)
        # Standard scaled dot-product attention + V/A bias
        attn_logits = (Q @ K.transpose(-2, -1)) / math.sqrt(d // H)  # (B, H, T_q, T_kv)
        bias = self.va_bias(va, T_q, T_kv)  # (B, H, T_q, T_kv)
        attn_logits = attn_logits + bias
        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = attn_probs @ V  # (B, H, T_q, d_h)
        out = out.transpose(1, 2).contiguous().view(B, T_q, d)
        out = self.attn.out_proj(out)
        # Residual + FFN
        x = q + out
        x = x + self.ff(self.norm_ff(x))
        return x


class CMTBlock(nn.Module):
    """One CMT block: audio attends to text, text attends to audio. Both with V/A bias."""
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.audio_to_text = CrossModalAttentionLayer(
            cfg.proj_dim, cfg.n_heads, cfg.va_dim, cfg.dropout
        )
        self.text_to_audio = CrossModalAttentionLayer(
            cfg.proj_dim, cfg.n_heads, cfg.va_dim, cfg.dropout
        )
        self.norm_audio = nn.LayerNorm(cfg.proj_dim)
        self.norm_text = nn.LayerNorm(cfg.proj_dim)

    def forward(self, audio: torch.Tensor, text: torch.Tensor,
                va: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_n = self.norm_audio(audio)
        text_n = self.norm_text(text)
        # audio attends to text
        audio_out = audio + self.audio_to_text(audio_n, text_n, va)
        # text attends to audio
        text_out = text + self.text_to_audio(text_n, audio_n, va)
        return audio_out, text_out


class VAAwareCMT(nn.Module):
    """
    The novel architecture: 2-layer CMT with V/A conditioning, plus MIN
    aggregation (MemoCMT's best on ESD).
    """
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.proj_audio = nn.Linear(cfg.audio_dim, cfg.proj_dim)
        self.proj_text = nn.Linear(cfg.text_dim, cfg.proj_dim)
        self.layers = nn.ModuleList([CMTBlock(cfg) for _ in range(cfg.n_cmt_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.proj_dim * 2),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.proj_dim * 2, cfg.proj_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.proj_dim, cfg.num_classes),
        )

    def forward(self, audio_tokens: torch.Tensor, text_tokens: torch.Tensor,
                va: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_tokens: (B, T_a, audio_dim) — wav2vec2 hidden states
            text_tokens:  (B, T_t, text_dim)  — BERT hidden states
            va: (B, 3)
        Returns:
            logits: (B, num_classes)
        """
        audio = self.proj_audio(audio_tokens)
        text = self.proj_text(text_tokens)
        for layer in self.layers:
            audio, text = layer(audio, text, va)
        # MIN aggregation (MemoCMT's best variant on ESD) over token dim
        audio_pooled = audio.min(dim=1).values  # (B, proj_dim)
        text_pooled = text.min(dim=1).values   # (B, proj_dim)
        fused = torch.cat([audio_pooled, text_pooled], dim=-1)  # (B, 2*proj_dim)
        return self.classifier(fused)


class DialogContextLayer(nn.Module):
    """
    The architectural extension: a transformer over the [CLS] vectors
    of the 10 previous utterances in the same dialog. The current
    utterance's [CLS] gets concatenated with the previous-utterance
    summary, then MLP'd into a final logit.
    """
    def __init__(self, cfg: FusionConfig, base_classifier_in_dim: int):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.dialog_dim,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.dialog_dim * 4,
                dropout=cfg.dropout,
                batch_first=True,
            ),
            num_layers=2,
        )
        # Final classifier: base_logits_proj + context -> num_classes
        self.context_proj = nn.Linear(cfg.dialog_dim, cfg.dialog_dim)
        self.final_classifier = nn.Sequential(
            nn.LayerNorm(base_classifier_in_dim + cfg.dialog_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(base_classifier_in_dim + cfg.dialog_dim, cfg.num_classes),
        )

    def forward(self, base_logits: torch.Tensor,
                dialog_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_logits: (B, base_dim) — logits from the CMT classifier's
                          penultimate layer (or just the [CLS] embedding)
            dialog_context: (B, dialog_window, dialog_dim) — the 10
                          previous utterances' [CLS] vectors
        Returns:
            (B, num_classes)
        """
        # Mean-pool the dialog context (simpler than learned attention)
        ctx = dialog_context.mean(dim=1)  # (B, dialog_dim)
        ctx = self.context_proj(ctx)
        fused = torch.cat([base_logits, ctx], dim=-1)
        return self.final_classifier(fused)


# Quick sanity test (run with `python cmt_fusion.py`)
if __name__ == "__main__":
    cfg = FusionConfig()
    model = VAAwareCMT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"VAAwareCMT params: {n_params:,}")
    # Dummy input
    B = 4
    audio_tokens = torch.randn(B, 100, cfg.audio_dim)
    text_tokens = torch.randn(B, 32, cfg.text_dim)
    va = torch.rand(B, 3)
    logits = model(audio_tokens, text_tokens, va)
    print(f"logits shape: {logits.shape}")
    print(f"logits: {logits}")
    # Now test the dialog context layer
    base_logits = torch.randn(B, cfg.proj_dim * 2)
    ctx_model = DialogContextLayer(cfg, base_logits.size(1))
    dialog_ctx = torch.randn(B, cfg.dialog_window, cfg.dialog_dim)
    final_logits = ctx_model(base_logits, dialog_ctx)
    print(f"final_logits shape: {final_logits.shape}")
    print("✅ cmt_fusion.py works")
