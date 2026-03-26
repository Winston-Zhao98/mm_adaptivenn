"""
training/curriculum.py

Three-phase curriculum learning scheduler.

Phase I   (0 → warmup_fraction):
    Only Steps 1 + 2A are active. RL gradients not yet introduced.
    Encoder and language encoder build stable task-relevant representations.

Phase II  (warmup_fraction → anneal_fraction):
    Steps 2B and 3 activate. α_B ramps from 0 → target.
    π^L variance σ cosine-anneals from σ_init → σ_final.
    Entropy coefficient λ_H linearly decays from max → final.

Phase III (anneal_fraction → 1.0):
    All steps active at target learning rates. σ and α_B fixed.

Reference: phase_b1_training.docx §5
"""
import math
import torch


class CurriculumScheduler:
    """
    Tracks training progress and adjusts:
      - which loss terms are active (phase I vs II/III)
      - α_B (language Path B learning rate multiplier)
      - σ_target (π^L variance target, fed to LocationAttentionPolicy)
      - entropy coefficient λ_H
    """

    def __init__(self, cfg):
        self.n_total   = cfg.training.n_total_steps
        self.warmup_n  = int(cfg.training.warmup_fraction  * self.n_total)
        self.anneal_n  = int(cfg.training.anneal_fraction  * self.n_total)

        self.sigma_init  = cfg.model.sigma_init
        self.sigma_final = cfg.model.sigma_final
        self.sigma_max   = cfg.model.sigma_max

        self.lr_lang_b_target = cfg.training.lr_lang_b
        self.entropy_init     = cfg.training.entropy_reg
        self.entropy_final    = cfg.training.entropy_final

        self.step = 0

    # ── Phase query ──────────────────────────────────────────────────────────

    @property
    def phase(self) -> int:
        if self.step < self.warmup_n:
            return 1
        elif self.step < self.anneal_n:
            return 2
        else:
            return 3

    @property
    def rl_active(self) -> bool:
        """Steps 2B and 3 are active from Phase II onwards."""
        return self.phase >= 2

    # ── Annealed quantities ───────────────────────────────────────────────────

    @property
    def alpha_b_multiplier(self) -> float:
        """
        Linear ramp of α_B from 0 to 1.0 during Phase II.
        Phase I: 0.0   Phase II: 0 → 1.0   Phase III: 1.0
        """
        if self.phase == 1:
            return 0.0
        elif self.phase == 3:
            return 1.0
        else:
            t = (self.step - self.warmup_n) / max(self.anneal_n - self.warmup_n, 1)
            return float(min(t, 1.0))

    @property
    def sigma_target(self) -> float:
        """
        Cosine annealing of π^L variance target during Phase II.
        σ: σ_init → σ_final
        """
        if self.phase == 1:
            return self.sigma_init
        elif self.phase == 3:
            return self.sigma_final
        else:
            t = (self.step - self.warmup_n) / max(self.anneal_n - self.warmup_n, 1)
            # Cosine annealing
            cosine = (1 + math.cos(math.pi * t)) / 2
            return self.sigma_final + (self.sigma_init - self.sigma_final) * cosine

    @property
    def entropy_coef(self) -> float:
        """
        Linear decay of entropy regularisation coefficient.
        λ_H: entropy_init → entropy_final
        """
        t = min(self.step / max(self.n_total, 1), 1.0)
        return self.entropy_init + (self.entropy_final - self.entropy_init) * t

    @property
    def effective_lr_b(self) -> float:
        return self.lr_lang_b_target * self.alpha_b_multiplier

    # ── Step ─────────────────────────────────────────────────────────────────

    def advance(self):
        self.step += 1

    def state_dict(self) -> dict:
        return {'step': self.step}

    def load_state_dict(self, d: dict):
        self.step = d['step']

    def summary(self) -> str:
        return (f"[Step {self.step}/{self.n_total}] "
                f"Phase {self.phase} | "
                f"RL active: {self.rl_active} | "
                f"α_B mult: {self.alpha_b_multiplier:.3f} | "
                f"σ_target: {self.sigma_target:.4f} | "
                f"λ_H: {self.entropy_coef:.4f}")
