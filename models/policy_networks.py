"""
models/policy_networks.py

Hierarchical policy networks π^M and π^L, Shared MLP, task head q, value V^π.

C4  / AC-2: All policy inputs use sg(s_{t-1}) — caller must pass detached state.
C5  / AC-4: θ_π^M ∩ θ_π^L = ∅ (separate nn.Linear output layers).
             θ_shared is a third, distinct parameter group.

Key design: z is injected ONCE via FiLM at the Shared MLP input.
            π^M and π^L receive the z-conditioned representation h.
            This is consistent with arch_03_policy.docx and Corollary 1.

Unified heterogeneous location space:
  π^L always outputs 3D params (μ_x, μ_y, μ_τ).
  At inference, sub-vector is selected based on m_t:
    vision (m=1) → (μ_x, μ_y, σ_x², σ_y²)
    audio  (m=2) → (μ_τ, σ_τ²)

Corresponds to arch_03_policy.docx.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Tuple, Dict, List, Optional

from .language_encoder import FiLMLayer


# ── Shared MLP ────────────────────────────────────────────────────────────────

class SharedMLP(nn.Module):
    """
    h = MLP_shared( FiLM(sg(s_{t-1}), z) )

    Parameters θ_shared are distinct from θ_π^M and θ_π^L.
    This module jointly feeds both policy heads.

    AC-2: caller passes DETACHED s (sg(s_{t-1})).
    """
    def __init__(self, d_s: int, d_z: int, d_h: int, n_layers: int = 3):
        super().__init__()
        self.film = FiLMLayer(d_z=d_z, d_h=d_s)

        layers = []
        in_dim = d_s
        for i in range(n_layers):
            out_dim = d_h if i == n_layers - 1 else d_h
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim)])
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.d_h = d_h

    def forward(self, s_sg: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        s_sg: (B, d_s) — MUST be sg(s_{t-1}), i.e. detached (AC-2)
        z:    (B, d_z) — fixed language context (AC-5)
        Returns h: (B, d_h)
        """
        conditioned = self.film(s_sg, z)                   # FiLM(sg(s), z)
        return self.mlp(conditioned)


# ── π^M: modality selection ───────────────────────────────────────────────────

class ModalitySelectionPolicy(nn.Module):
    """
    π^M: Categorical distribution over ℳ.
    m_t ~ Categorical(π^M(h))

    Parameters θ_π^M = {W_M, b_M}.
    Trained via REINFORCE with advantage A_t.
    When |ℳ|=1, always returns the single modality (degenerate case).
    """
    def __init__(self, d_h: int, modalities: List[int]):
        super().__init__()
        self.modalities = modalities
        self.M = len(modalities)
        # θ_π^M: disjoint from θ_π^L
        self.head = nn.Linear(d_h, self.M)

    def forward(self, h: torch.Tensor) -> Categorical:
        """Returns a Categorical distribution over modalities."""
        logits = self.head(h)                              # (B, |ℳ|)
        return Categorical(logits=logits)

    def select(self, h: torch.Tensor,
               greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (m_t_indices, log_probs).
        m_t_indices: (B,) integer indices into self.modalities list
        """
        dist = self.forward(h)
        if greedy or self.M == 1:
            m_idx = dist.probs.argmax(dim=-1)
        else:
            m_idx = dist.sample()
        log_probs = dist.log_prob(m_idx)
        return m_idx, log_probs

    def modality_ids(self, m_idx: torch.Tensor) -> torch.Tensor:
        """Convert index in [0, M) to actual modality ID (1 or 2)."""
        ids = torch.tensor(self.modalities, device=m_idx.device)
        return ids[m_idx]


# ── π^L: location attention ───────────────────────────────────────────────────

class LocationAttentionPolicy(nn.Module):
    """
    π^L: Gaussian distribution over location.

    Unified 3D parameterisation: always outputs (μ_x, μ_y, μ_τ) and (σ_x², σ_y², σ_τ²).
    Sub-vector selection based on m_t at inference:
      vision (m=1): use (μ_x, μ_y, σ_x², σ_y²)
      audio  (m=2): use (μ_τ, σ_τ²)

    Continuous action → trained via reparameterisation trick.
    Parameters θ_π^L = {W_μ, b_μ, W_σ, b_σ}.
    """
    def __init__(self, d_h: int, sigma_max: float = 0.3):
        super().__init__()
        self.sigma_max = sigma_max
        # θ_π^L: disjoint from θ_π^M
        self.mu_head = nn.Linear(d_h, 3)                  # (μ_x, μ_y, μ_τ)
        self.sigma_head = nn.Linear(d_h, 3)               # (σ_x², σ_y², σ_τ²)
        # Initialise sigma_head bias to a negative value so that
        # softplus(bias) ≈ 0.05 < sigma_max at the start of training.
        # Without this, default init produces outputs >> sigma_max,
        # the clamp activates everywhere, and the gradient is zero
        # throughout — breaking policy gradient at training step 0.
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, -2.9)     # softplus(-2.9) ≈ 0.055

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mu, sigma): each (B, 3)
        mu   ∈ (0, 1)^3  via Sigmoid
        sigma ∈ (0, sigma_max]  via Softplus + clamp
        """
        mu = torch.sigmoid(self.mu_head(h))                # (B, 3), ∈ (0,1)
        sigma = F.softplus(self.sigma_head(h))
        sigma = sigma.clamp(max=self.sigma_max)
        return mu, sigma

    def select(self, h: torch.Tensor,
               m_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterisation: l_t = μ + σ * ε, ε ~ N(0,I).
        m_ids: (B,) actual modality IDs (1 for vision, 2 for audio)

        Returns:
          l_t:      (B, max_dim) — 2D for vision rows, 1D for audio rows
                    padded with zeros for mismatched dims
          log_prob: (B,) log probability under the selected Gaussian
        """
        mu, sigma = self.forward(h)                        # (B, 3)

        # Reparameterise
        eps = torch.randn_like(mu)
        l_full = (mu + sigma * eps).clamp(0., 1.)          # (B, 3)

        # Per-sample sub-vector selection and log-prob computation
        log_probs = []
        l_selected = []

        for b in range(m_ids.shape[0]):
            m = m_ids[b].item()
            if m == 1:                                     # vision: 2D
                l_b = l_full[b, :2]                        # (μ_x, μ_y)
                dist = Normal(mu[b, :2], sigma[b, :2])
                lp = dist.log_prob(l_b).sum()
            else:                                          # audio: 1D
                l_b = l_full[b, 2:3]                       # (μ_τ,)
                dist = Normal(mu[b, 2:3], sigma[b, 2:3])
                lp = dist.log_prob(l_b).sum()
                l_b = F.pad(l_b, (0, 1))                   # pad to length 2

            l_selected.append(l_b)
            log_probs.append(lp)

        l_t = torch.stack(l_selected, dim=0)               # (B, 2)
        log_prob = torch.stack(log_probs, dim=0)           # (B,)
        return l_t, log_prob

    def entropy(self, h: torch.Tensor) -> torch.Tensor:
        """Gaussian entropy for regularisation."""
        _, sigma = self.forward(h)
        # H[N(μ,σ²)] = 0.5 * log(2πe σ²) = 0.5 * (1 + log(2π) + 2*log(σ))
        # Use math.log to get a Python float — avoids CPU/CUDA device mismatch.
        return 0.5 * (1 + math.log(2 * math.pi) + 2 * sigma.log()).sum(-1)


# ── Task head q ───────────────────────────────────────────────────────────────

class TaskHead(nn.Module):
    """
    q: ŷ_t = q(s_t, z)
    Fuses perceptual state s_t with language context z via concatenation or cross-attention.
    Parameters θ_q are updated in Step 1 (representation learning).

    AC-1: z is injected HERE (not upstream in Ψ). s_t comes from full graph.
    """
    def __init__(self, d_s: int, d_z: int, num_classes: int,
                 fusion: str = 'concat'):
        super().__init__()
        self.fusion = fusion

        if fusion == 'concat':
            self.classifier = nn.Sequential(
                nn.Linear(d_s + d_z, d_s),
                nn.GELU(),
                nn.LayerNorm(d_s),
                nn.Linear(d_s, num_classes),
            )
        elif fusion == 'cross_attn':
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_s, num_heads=8, batch_first=True
            )
            self.z_proj = nn.Linear(d_z, d_s)
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_s),
                nn.Linear(d_s, num_classes),
            )
        self.num_classes = num_classes

    def forward(self, s_t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        s_t: (B, d_s) — from FULL computation graph (Step 1 uses BPTT through s_t)
        z:   (B, d_z) — fixed language context (AC-5)
        Returns logits: (B, num_classes)
        """
        if self.fusion == 'concat':
            fused = torch.cat([s_t, z], dim=-1)
            return self.classifier(fused)
        else:
            z_key = self.z_proj(z).unsqueeze(1)            # (B, 1, d_s)
            q_val = s_t.unsqueeze(1)                       # (B, 1, d_s)
            out, _ = self.cross_attn(q_val, z_key, z_key)
            return self.classifier(out.squeeze(1))


# ── Value network V^π ─────────────────────────────────────────────────────────

class ValueNetwork(nn.Module):
    """
    V^π: estimates expected return from state s_{t-1}.
    Input: sg(s_{t-1}) — always detached.
    Updated in Step 4 via MSE against Monte-Carlo returns R_t.
    Parameters θ_{V^π} are separate from all policy parameters.
    """
    def __init__(self, d_s: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_s, d_s // 2),
            nn.GELU(),
            nn.LayerNorm(d_s // 2),
            nn.Linear(d_s // 2, 1),
        )

    def forward(self, s_sg: torch.Tensor) -> torch.Tensor:
        """
        s_sg: (B, d_s) — MUST be detached (sg(s_{t-1}))
        Returns: (B,) estimated value
        """
        return self.net(s_sg).squeeze(-1)
