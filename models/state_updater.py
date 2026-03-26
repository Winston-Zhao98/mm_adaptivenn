"""
models/state_updater.py

State updater Ψ: s_t = Ψ(s_{t-1}, o_t)

C2  / AC-1: Ψ receives ONLY (s_{t-1}, o_t). z is NEVER an input.
             This is the critical constraint for Proposition 2.
C4  / AC-2: During RL/policy updates, s_{t-1} must be detached BEFORE
             passing to Ψ. The model itself does NOT detach — the training
             loop is responsible for passing detach(s_{t-1}) when needed.

Three backbone variants: GRU (main), Transformer/causal (ablation A7), LSTM.
All have identical interface: forward(s_prev, o_t) → s_t.

Corresponds to arch_02_psi_lang.docx §1–§3.
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class GRUStateUpdater(nn.Module):
    """
    Ψ-GRU: single-layer Gated Recurrent Unit.
    Input: concatenated [s_{t-1}, o_t] projected to d_s.
    """
    def __init__(self, d: int, d_s: int):
        super().__init__()
        # GRU input is o_t (dim d), hidden state is s (dim d_s)
        self.gru_cell = nn.GRUCell(input_size=d, hidden_size=d_s)
        self.d_s = d_s
        self.d = d

    def initial_state(self, batch_size: int,
                      device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.d_s, device=device)

    def forward(self, s_prev: torch.Tensor,
                o_t: torch.Tensor) -> torch.Tensor:
        """
        s_prev: (B, d_s) — may be detached by caller (AC-2)
        o_t:    (B, d)   — no z here (AC-1)
        Returns s_t: (B, d_s)
        """
        return self.gru_cell(o_t, s_prev)


class LSTMStateUpdater(nn.Module):
    """
    Ψ-LSTM: single-layer Long Short-Term Memory.
    State is (h_t, c_t); we concatenate them as the 'state' vector s_t.
    """
    def __init__(self, d: int, d_s: int):
        super().__init__()
        assert d_s % 2 == 0, "d_s must be even for LSTM (split into h and c)"
        self.d_h = d_s // 2
        self.lstm_cell = nn.LSTMCell(input_size=d, hidden_size=self.d_h)
        self.d_s = d_s
        self.d = d

    def initial_state(self, batch_size: int,
                      device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.d_s, device=device)

    def _split(self, s: torch.Tensor):
        return s[:, :self.d_h], s[:, self.d_h:]

    def forward(self, s_prev: torch.Tensor,
                o_t: torch.Tensor) -> torch.Tensor:
        h_prev, c_prev = self._split(s_prev)
        h_t, c_t = self.lstm_cell(o_t, (h_prev, c_prev))
        return torch.cat([h_t, c_t], dim=1)


class CausalTransformerStateUpdater(nn.Module):
    """
    Ψ-Trans: causal Transformer that accumulates the observation history.
    State s_t is the mean-pooled representation of all observations o_1…o_t.
    Maintains a sequence buffer internally.

    NOTE: For the perception loop, we operate step-by-step.
          At step t, we have access to o_1, …, o_t.
          We recompute the representation from scratch each step
          (or use an incremental cache for efficiency).
    """
    def __init__(self, d: int, d_s: int, n_layers: int = 4, n_heads: int = 8,
                 max_seq_len: int = 10):
        super().__init__()
        self.d_s = d_s
        self.d = d
        self.max_seq_len = max_seq_len

        # Input projection: o_t (dim d) → d_s
        self.input_proj = nn.Linear(d, d_s)

        # Positional encoding
        self.pos_emb = nn.Embedding(max_seq_len, d_s)

        # Causal Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_s, nhead=n_heads, dim_feedforward=d_s * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_s)

    def initial_state(self, batch_size: int,
                      device: torch.device) -> torch.Tensor:
        # State is empty history — represented as zeros
        return torch.zeros(batch_size, self.d_s, device=device)

    def _causal_mask(self, seq_len: int,
                     device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, s_prev: torch.Tensor,
                o_t: torch.Tensor,
                history: Optional[torch.Tensor] = None,
                step: int = 0) -> torch.Tensor:
        """
        For the causal Transformer, we need the full history.
        `history` is (B, t, d) — all observations so far including o_t.
        If history is None, we treat o_t as the only observation.

        Returns s_t: (B, d_s)  mean-pooled over history.
        """
        if history is None:
            # First step or no history: just encode o_t
            history = o_t.unsqueeze(1)                     # (B, 1, d)

        B, L, _ = history.shape
        tokens = self.input_proj(history)                  # (B, L, d_s)
        pos = torch.arange(L, device=history.device)
        tokens = tokens + self.pos_emb(pos).unsqueeze(0)

        mask = self._causal_mask(L, history.device)
        out = self.transformer(tokens, mask=mask)
        out = self.norm(out)
        return out[:, -1, :]                               # last token as state


# ── Factory ───────────────────────────────────────────────────────────────────

def build_state_updater(cfg) -> nn.Module:
    d, d_s = cfg.model.d, cfg.model.d_s
    if cfg.model.psi_backbone == 'gru':
        return GRUStateUpdater(d=d, d_s=d_s)
    elif cfg.model.psi_backbone == 'lstm':
        return LSTMStateUpdater(d=d, d_s=d_s)
    elif cfg.model.psi_backbone == 'transformer':
        return CausalTransformerStateUpdater(
            d=d, d_s=d_s,
            n_layers=cfg.model.psi_transformer_layers,
            n_heads=cfg.model.psi_transformer_heads,
        )
    else:
        raise ValueError(f"Unknown Ψ backbone: {cfg.model.psi_backbone}")
