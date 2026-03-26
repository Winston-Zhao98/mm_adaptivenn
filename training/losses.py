"""
training/losses.py

All loss terms implementing Theorem 1's gradient decomposition:

  ∇_θ L(θ) = ∇_θ L_rep(θ) + ∇_θ L_lang(θ) + ∇_θ L_rl(θ)

  L_rep:   representation learning  → {θ_rep, θ_Ψ, θ_q}   full BPTT
  L_lang:  language encoder         → θ_lang               dual path A+B
  L_rl:    self-rewarding RL        → {θ_π^M, θ_π^L, θ_shared}  REINFORCE + reparam
  L_align: cross-modal alignment    → {θ_rep^(1), θ_rep^(2)}    auxiliary

Reference: theorem_final.docx, phase_b1_training.docx §1
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional


def compute_stop_weighted_loss(
    y_hat: torch.Tensor,       # (B, T, num_classes)
    y: torch.Tensor,           # (B,) integer labels
    stop_probs: torch.Tensor,  # (T,) P(t_o = t)
) -> torch.Tensor:
    """
    L_rep / task loss weighted by stopping distribution:
        L = Σ_t P(t_o=t) · CE(y, ŷ_t)

    Uses full computation graph (y_hat comes from s_full chain).
    """
    B, T, C = y_hat.shape
    loss = 0.
    for t in range(T):
        ce_t = F.cross_entropy(y_hat[:, t, :], y, reduction='mean')
        loss = loss + stop_probs[t] * ce_t
    return loss


def compute_rewards(
    y_hat: torch.Tensor,       # (B, T, num_classes)
    y: torch.Tensor,           # (B,)
    stop_probs: torch.Tensor,  # (T,)
    gamma: float = 0.99,
) -> Dict[str, torch.Tensor]:
    """
    Self-rewarding signal R_t = Σ_{t'≥t} γ^{t'-t} r_{t'}.
    r_t = -P(t_o=t) · CE(y, ŷ_t)

    Returns dict with:
      rewards:  (B, T)  r_t per step
      returns:  (B, T)  R_t = Σ_{t'≥t} γ^{t'-t} r_{t'} (reward-to-go)
    """
    B, T, C = y_hat.shape

    # Compute per-step rewards (detached — reward is a scalar target)
    with torch.no_grad():
        rewards = torch.zeros(B, T, device=y_hat.device)
        for t in range(T):
            ce_t = F.cross_entropy(
                y_hat[:, t, :].detach(), y, reduction='none'
            )                                               # (B,)
            rewards[:, t] = -stop_probs[t] * ce_t

        # Reward-to-go: R_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        returns = torch.zeros_like(rewards)
        running = torch.zeros(B, device=y_hat.device)
        for t in reversed(range(T)):
            running = rewards[:, t] + gamma * running
            returns[:, t] = running

    return {'rewards': rewards, 'returns': returns}


def compute_advantages(
    returns: torch.Tensor,     # (B, T)
    values: torch.Tensor,      # (B, T) V^π(sg(s_{t-1}))
    normalise: bool = True,
) -> torch.Tensor:
    """
    A_t = R_t - V^π(sg(s_{t-1}))

    Optionally normalise per batch for variance reduction.
    """
    advantages = returns - values.detach()                  # detach value baseline
    if normalise:
        mean = advantages.mean()
        std  = advantages.std().clamp(min=1e-8)
        advantages = (advantages - mean) / std
    return advantages                                        # (B, T)


# ── Step 1: L_rep ─────────────────────────────────────────────────────────────

def loss_rep(
    y_hat: torch.Tensor,       # (B, T, C) — from s_full chain (full graph)
    y: torch.Tensor,           # (B,)
    stop_probs: torch.Tensor,  # (T,)
) -> torch.Tensor:
    """
    ∇_θ L_rep: governs {θ_rep, θ_Ψ, θ_q}.
    Full BPTT — y_hat must come from s_full (not s_sg).
    """
    return compute_stop_weighted_loss(y_hat, y, stop_probs)


# ── Step 2: L_lang ────────────────────────────────────────────────────────────

def loss_lang_path_a(
    y_hat: torch.Tensor,       # (B, T, C) — full graph (z → task head → y_hat)
    y: torch.Tensor,
    stop_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Path A: ∇_{θ_lang} from task accuracy (direct gradient z → q → y_hat).
    Same computation as L_rep but used to update θ_lang with lr_lang_a.
    C6: ∂s_t/∂θ_lang = 0, so gradient flows ONLY through z → q path.
    """
    return compute_stop_weighted_loss(y_hat, y, stop_probs)


def loss_lang_path_b(
    log_pi: torch.Tensor,      # (B, T) log p(a_t | sg(s_{t-1}), z)
    advantages: torch.Tensor,  # (B, T)
) -> torch.Tensor:
    """
    Path B: ∇_{θ_lang} from policy quality (REINFORCE through z → π).
    Uses sg(s_{t-1}) in policy, so gradient flows ONLY through z → FiLM → h → π.

    L_lang^B = -E[Σ_t A_t · ∇_{θ_lang} log p(a_t | sg(s), z)]
    """
    # REINFORCE: maximise E[A_t · log p(a_t)]
    return -(advantages.detach() * log_pi).mean()


# ── Step 3: L_rl ──────────────────────────────────────────────────────────────

def loss_rl(
    log_pi: torch.Tensor,      # (B, T) log p(a_t) = log_pi_M + log_pi_L
    advantages: torch.Tensor,  # (B, T)
    entropy: Optional[torch.Tensor] = None,   # (B, T) π^L entropy
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    """
    L_rl: governs {θ_π^M, θ_π^L, θ_shared}.
    REINFORCE for π^M (discrete) + reparameterisation for π^L (continuous).
    Combined as: L_rl = -E[Σ_t A_t · log p(a_t)]

    Entropy regularisation encourages exploration during early training.
    """
    policy_loss = -(advantages.detach() * log_pi).mean()

    if entropy is not None and entropy_coef > 0:
        entropy_bonus = -entropy_coef * entropy.mean()
        return policy_loss + entropy_bonus

    return policy_loss


# ── Step 4: L_value ───────────────────────────────────────────────────────────

def loss_value(
    values: torch.Tensor,      # (B, T) V^π(sg(s_{t-1})) — from s_sg chain
    returns: torch.Tensor,     # (B, T) R_t (Monte-Carlo, detached)
) -> torch.Tensor:
    """
    MSE regression: L_value = E[(V^π(sg(s)) - R_t)²]
    """
    return F.mse_loss(values, returns.detach())


# ── Auxiliary: L_align ────────────────────────────────────────────────────────

def loss_align(
    o_vision: torch.Tensor,    # (B, d) vision features
    o_audio: torch.Tensor,     # (B, d) audio features
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Cross-modal contrastive alignment loss (auxiliary, λ_align = 0.1).
    Encourages visual and auditory features of semantically related pairs
    to be close in ℝ^d.

    j sums over ALL samples in the mini-batch (positive pair included in denominator).
    Only updates {θ_rep^(1), θ_rep^(2)} — does not affect Theorem 1 decomposition.
    """
    # Normalise
    v = F.normalize(o_vision, dim=-1)                      # (B, d)
    a = F.normalize(o_audio, dim=-1)                       # (B, d)

    # Similarity matrix
    sim = torch.mm(v, a.T) / temperature                   # (B, B)

    # Symmetric contrastive loss
    labels = torch.arange(sim.shape[0], device=sim.device)
    loss_v2a = F.cross_entropy(sim, labels)
    loss_a2v = F.cross_entropy(sim.T, labels)
    return (loss_v2a + loss_a2v) / 2
