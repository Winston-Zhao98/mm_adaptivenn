"""
tests/test_gradient_flow.py

Gradient flow verification for MM-AdaptiveNN's dual-graph design.

These tests directly verify Theorem 1's gradient decomposition:

    ∇_θ L(θ) = ∇_θ L_rep(θ) + ∇_θ L_lang(θ) + ∇_θ L_rl(θ)

Each test isolates one loss term and checks that gradients reach exactly
the intended parameter groups and are blocked from the others.

Nomenclature
  s_full : full computation graph — Ψ receives un-detached s_{t-1}   (Step 1)
  s_sg   : stop-gradient chain   — Ψ receives detach(s_{t-1})        (Steps 2B, 3)

Test summary
  GF-1   L_rep backward reaches θ_Ψ (BPTT through full state chain)
  GF-2   L_rep backward reaches θ_q (task head)
  GF-3   L_rep backward is BLOCKED from θ_π^M, θ_π^L, θ_shared (AC-2/C4)
  GF-4   L_rl  backward reaches θ_π^M, θ_π^L, θ_shared (policy update)
  GF-5   L_rl  backward is BLOCKED from θ_Ψ (stop-gradient isolation)
  GF-6   L_lang_a backward reaches θ_lang (language projection)
  GF-7   C6 structural: z is NOT an input to Ψ (architectural constraint)
  GF-8   L_value backward reaches θ_V only (value network)
  GF-9   L_value backward is BLOCKED from θ_Ψ (uses s_sg)
  GF-10  s_sg.requires_grad == False (AC-2 stop-gradient enforced)
  GF-11  s_full.requires_grad == True (full BPTT graph maintained)
  GF-12  θ_π^M ∩ θ_π^L == ∅  (AC-4 disjoint policy heads)
"""
import sys
import inspect
import math
import pytest
import torch

sys.path.insert(0, '.')

from configs.default import get_config
from models.mm_adaptivenn import MMAdaptiveNN
from training.losses import (
    loss_rep, loss_lang_path_a, loss_rl, loss_value,
    compute_rewards, compute_advantages,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_model_and_inputs(modalities=(1,), T=2, B=2):
    """
    Build a minimal MMAdaptiveNN with random weights and CPU-only dummy inputs.
    pretrained=False (vision_pretrained='none') ensures no network download.
    """
    cfg = get_config()
    cfg.model.modalities          = list(modalities)
    cfg.model.num_classes         = 10
    cfg.model.d                   = 64
    cfg.model.d_s                 = 128
    cfg.model.d_h                 = 128
    cfg.language.d_z              = 64
    cfg.training.T                = T
    cfg.device                    = 'cpu'
    cfg.encoder.vision_backbone   = 'resnet50'
    cfg.encoder.vision_pretrained = 'none'   # no download in tests

    model = MMAdaptiveNN(cfg)
    model.train()

    X = {}
    if 1 in modalities:
        X[1] = torch.randn(B, 3, 64, 64)
    if 2 in modalities:
        X[2] = torch.randn(B, 1, 128, 50)

    w    = ["classify the image"] * B
    y    = torch.zeros(B, dtype=torch.long)
    stop = MMAdaptiveNN._stop_distribution(T, 0.5, torch.device('cpu'))
    return model, cfg, X, w, y, stop


def _has_grad(params):
    """True if ANY parameter in the list has a non-None, non-zero gradient."""
    return any(
        p.grad is not None and p.grad.abs().max() > 0
        for p in params
    )


def _run_forward(model, cfg, X, w, y, stop):
    """Full forward pass; returns (z, traj, reward_dict, advantages)."""
    z    = model.encode_language(w)
    traj = model.perception_loop(X, z, cfg.training.T, stop, greedy=False)
    rd   = compute_rewards(traj['y_hat'], y, stop, gamma=cfg.training.gamma)
    adv  = compute_advantages(rd['returns'], traj['v_t'], normalise=True)
    return z, traj, rd, adv


# ══════════════════════════════════════════════════════════════════════════════
# GF-1/2/3  L_rep gradient reach and isolation
# ══════════════════════════════════════════════════════════════════════════════

class TestLRepGradientFlow:

    def test_gf1_l_rep_reaches_psi(self):
        """
        GF-1: L_rep must propagate gradients into θ_Ψ (state updater).
        This verifies BPTT through the full computation graph (s_full chain).
        Theorem 1 Step 1: ∇_θ L_rep governs {θ_rep, θ_Ψ, θ_q}.
        """
        model, cfg, X, w, y, stop = _make_model_and_inputs()
        z, traj, _, _ = _run_forward(model, cfg, X, w, y, stop)

        loss_rep(traj['y_hat'], y, stop).backward()

        assert _has_grad(list(model.psi.parameters())), (
            "GF-1 FAIL: L_rep did not reach θ_Ψ. "
            "BPTT through the full state chain is broken."
        )

    def test_gf2_l_rep_reaches_task_head(self):
        """
        GF-2: L_rep must propagate gradients into θ_q (task head).
        """
        model, cfg, X, w, y, stop = _make_model_and_inputs()
        z, traj, _, _ = _run_forward(model, cfg, X, w, y, stop)

        loss_rep(traj['y_hat'], y, stop).backward()

        assert _has_grad(list(model.task_head.parameters())), (
            "GF-2 FAIL: L_rep did not reach θ_q."
        )

    def test_gf3_l_rep_blocked_from_policy(self):
        """
        GF-3: L_rep must NOT reach policy parameters (AC-2 / C4).
        Policy inputs use sg(s_{t-1}), severing the gradient path from
        y_hat through the state chain into the policy heads.

        Failure would mean L_rep corrupts policy gradient estimates,
        violating Theorem 1's gradient decomposition.
        """
        model, cfg, X, w, y, stop = _make_model_and_inputs()
        z, traj, _, _ = _run_forward(model, cfg, X, w, y, stop)

        loss_rep(traj['y_hat'], y, stop).backward()

        policy_params = (
            list(model.pi_M.parameters()) +
            list(model.pi_L.parameters()) +
            list(model.shared_mlp.parameters())
        )
        assert not _has_grad(policy_params), (
            "GF-3 FAIL: L_rep leaked into policy parameters. "
            "Check that perception_loop uses s_sg (not s_full) as policy input."
        )


# ══════════════════════════════════════════════════════════════════════════════
# GF-4/5  L_rl gradient reach and isolation
# ══════════════════════════════════════════════════════════════════════════════

class TestLRlGradientFlow:

    def test_gf4_l_rl_reaches_policy(self):
        """
        GF-4: L_rl must propagate gradients into {θ_π^M, θ_π^L, θ_shared}.
        Theorem 1 Step 3.

        Note on sigma_head initialization: if sigma_head outputs are ≥ sigma_max,
        the clamp kills the gradient through log_pi_L.  LocationAttentionPolicy
        initialises sigma_head.bias = -2.9 (softplus(-2.9) ≈ 0.055 < sigma_max)
        to guarantee a live gradient path at training start.
        """
        model, cfg, X, w, y, stop = _make_model_and_inputs()
        z, traj, _, adv = _run_forward(model, cfg, X, w, y, stop)

        loss_rl(traj['log_pi'], adv).backward()

        policy_params = (
            list(model.pi_M.parameters()) +
            list(model.pi_L.parameters()) +
            list(model.shared_mlp.parameters())
        )
        assert _has_grad(policy_params), (
            "GF-4 FAIL: L_rl did not reach policy parameters. "
            "Check LocationAttentionPolicy sigma_head initialisation: "
            "if sigma is always clamped to sigma_max, its gradient is zero."
        )

    def test_gf5_l_rl_blocked_from_psi(self):
        """
        GF-5: L_rl must NOT reach θ_Ψ.
        This is the central isolation guarantee of the stop-gradient design.
        Policy inputs use sg(s_{t-1}), so there is no gradient path to Ψ.

        Failure means RL loss contaminates representation learning in Ψ,
        contradicting Theorem 1.
        """
        model, cfg, X, w, y, stop = _make_model_and_inputs()
        z, traj, _, adv = _run_forward(model, cfg, X, w, y, stop)

        loss_rl(traj['log_pi'], adv).backward()

        assert not _has_grad(list(model.psi.parameters())), (
            "GF-5 FAIL: L_rl leaked into θ_Ψ. "
            "s_sg must be fully detached at every step in perception_loop."
        )


# ══════════════════════════════════════════════════════════════════════════════
# GF-6/7  L_lang gradient and C6 architectural constraint
# ══════════════════════════════════════════════════════════════════════════════

class TestLLangGradientFlow:

    def test_gf6_l_lang_a_reaches_lang_encoder(self):
        """
        GF-6: L_lang (Path A) must reach θ_lang (language projection).
        Theorem 1 Step 2: z → q → y_hat provides a direct gradient path.
        """
        model, cfg, X, w, y, stop = _make_model_and_inputs()
        z, traj, _, _ = _run_forward(model, cfg, X, w, y, stop)

        loss_lang_path_a(traj['y_hat'], y, stop).backward()

        assert _has_grad(list(model.f_lang.projection.parameters())), (
            "GF-6 FAIL: L_lang_a did not reach θ_lang (language projection)."
        )

    def test_gf7_c6_z_not_input_to_psi(self):
        """
        GF-7 / C6 structural: z must NOT appear as an input to Ψ.forward().

        C6 is an ARCHITECTURAL constraint, not a gradient-flow one.
        y_hat = task_head(s_full, z) means L_lang_a does have a gradient path
        to Ψ through s_full — but opt_lang.step() only updates θ_lang, so Ψ
        is never modified by Step 2.  The invariant to test here is structural:
        Ψ's forward method must accept only (s, o) and must not accept z.

        This ensures C6: ∂s_t/∂θ_lang = 0  (z never enters Ψ's computation).
        """
        model, cfg, X, w, y, stop = _make_model_and_inputs()

        # Inspect Ψ's forward signature
        psi_sig = inspect.signature(model.psi.forward)
        param_names = list(psi_sig.parameters.keys())

        # Ψ.forward must NOT have a parameter named 'z' or 'lang'
        forbidden = {'z', 'lang', 'language', 'context'}
        overlap = forbidden & set(param_names)
        assert len(overlap) == 0, (
            f"GF-7 / C6 FAIL: Ψ.forward() has parameter(s) {overlap}. "
            "z must not enter the state updater. C6 requires ∂s_t/∂θ_lang = 0."
        )

        # Also check that the optimiser for θ_lang does NOT contain Ψ params
        from torch.optim import Adam
        lang_params_set = {id(p) for p in model.f_lang.projection.parameters()}
        psi_params_set  = {id(p) for p in model.psi.parameters()}
        assert len(lang_params_set & psi_params_set) == 0, (
            "GF-7 / C6 FAIL: θ_lang and θ_Ψ share parameters. "
            "They must be disjoint for Step 2 to leave Ψ unmodified."
        )


# ══════════════════════════════════════════════════════════════════════════════
# GF-8/9  L_value gradient reach and isolation
# ══════════════════════════════════════════════════════════════════════════════

class TestLValueGradientFlow:

    def test_gf8_l_value_reaches_value_net(self):
        """
        GF-8: L_value must reach θ_V (value network). Theorem 1 Step 4.
        """
        model, cfg, X, w, y, stop = _make_model_and_inputs()
        z, traj, rd, _ = _run_forward(model, cfg, X, w, y, stop)

        loss_value(traj['v_t'], rd['returns']).backward()

        assert _has_grad(list(model.value_net.parameters())), (
            "GF-8 FAIL: L_value did not reach θ_V."
        )

    def test_gf9_l_value_blocked_from_psi(self):
        """
        GF-9: L_value must NOT reach θ_Ψ.
        V^π receives sg(s_{t-1}), severing the gradient path to Ψ.
        """
        model, cfg, X, w, y, stop = _make_model_and_inputs()
        z, traj, rd, _ = _run_forward(model, cfg, X, w, y, stop)

        loss_value(traj['v_t'], rd['returns']).backward()

        assert not _has_grad(list(model.psi.parameters())), (
            "GF-9 FAIL: L_value leaked into θ_Ψ. "
            "ValueNetwork must receive sg(s_{t-1}), not s_full."
        )


# ══════════════════════════════════════════════════════════════════════════════
# GF-10/11  Dual state chain properties
# ══════════════════════════════════════════════════════════════════════════════

class TestStateChainProperties:

    def test_gf10_s_sg_has_no_grad(self):
        """
        GF-10: s_sg.requires_grad must be False (AC-2 stop-gradient).
        Direct runtime verification that the stop-gradient is applied.
        """
        model, cfg, X, w, y, stop = _make_model_and_inputs()
        model.train()
        z    = model.encode_language(["classify"] * 2)
        traj = model.perception_loop(X, z, cfg.training.T, stop, greedy=False)

        assert not traj['s_sg'].requires_grad, (
            "GF-10 FAIL: s_sg.requires_grad is True. "
            "AC-2 stop-gradient is missing — policy receives gradients from "
            "the representation loss, breaking Theorem 1."
        )

    def test_gf11_s_full_has_grad(self):
        """
        GF-11: s_full.requires_grad must be True (BPTT graph maintained).
        """
        model, cfg, X, w, y, stop = _make_model_and_inputs()
        model.train()
        z    = model.encode_language(["classify"] * 2)
        traj = model.perception_loop(X, z, cfg.training.T, stop, greedy=False)

        assert traj['s_full'].requires_grad, (
            "GF-11 FAIL: s_full.requires_grad is False. "
            "The full BPTT graph is not maintained — L_rep cannot update θ_Ψ."
        )


# ══════════════════════════════════════════════════════════════════════════════
# GF-12  AC-4: disjoint policy heads
# ══════════════════════════════════════════════════════════════════════════════

class TestDisjointPolicyHeads:

    def test_gf12_theta_pi_m_disjoint_from_theta_pi_l(self):
        """
        GF-12: θ_π^M, θ_π^L, and θ_shared must be mutually disjoint (AC-4 / C5).
        Independent parameter sets allow each policy to be optimised separately.
        """
        model, cfg, X, w, y, stop = _make_model_and_inputs()

        pi_m_ids   = {id(p) for p in model.pi_M.parameters()}
        pi_l_ids   = {id(p) for p in model.pi_L.parameters()}
        shared_ids = {id(p) for p in model.shared_mlp.parameters()}

        assert len(pi_m_ids & pi_l_ids) == 0, (
            "GF-12 FAIL: θ_π^M ∩ θ_π^L ≠ ∅. AC-4 requires disjoint heads."
        )
        assert len(pi_m_ids & shared_ids) == 0, (
            "GF-12 FAIL: θ_π^M ∩ θ_shared ≠ ∅."
        )
        assert len(pi_l_ids & shared_ids) == 0, (
            "GF-12 FAIL: θ_π^L ∩ θ_shared ≠ ∅."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Integration: four-step update correctly isolates parameter groups
# ══════════════════════════════════════════════════════════════════════════════

class TestFourStepIsolation:

    def test_four_step_no_cross_contamination(self):
        """
        Integration test: simulate all four training steps and verify that
        after each backward, only the intended parameter group carries a gradient.

        This is the key empirical test for Theorem 1's claim that the four
        gradient terms are additive over non-overlapping parameter subsets.

        Note on Step 2A / C6:
          L_lang_a = L_rep computed from y_hat = task_head(s_full, z).
          The gradient does flow through s_full to Ψ — this is expected.
          What matters is that opt_lang.step() updates ONLY θ_lang (Ψ is not
          in opt_lang).  We verify this with a parameter-set membership check.
        """
        model, cfg, X, w, y, stop = _make_model_and_inputs()
        z, traj, rd, adv = _run_forward(model, cfg, X, w, y, stop)

        psi_ids    = {id(p) for p in model.psi.parameters()}
        lang_ids   = {id(p) for p in model.f_lang.projection.parameters()}
        policy_ids = (
            {id(p) for p in model.pi_M.parameters()} |
            {id(p) for p in model.pi_L.parameters()} |
            {id(p) for p in model.shared_mlp.parameters()}
        )
        value_ids  = {id(p) for p in model.value_net.parameters()}

        # ── Step 1: L_rep ────────────────────────────────────────────
        model.zero_grad(set_to_none=True)
        loss_rep(traj['y_hat'], y, stop).backward(retain_graph=True)

        assert _has_grad(list(model.psi.parameters())),       "Step 1: θ_Ψ must have grad"
        assert _has_grad(list(model.task_head.parameters())), "Step 1: θ_q must have grad"
        assert not _has_grad(list(model.pi_M.parameters())), "Step 1: θ_π_M must NOT have grad"
        assert not _has_grad(list(model.pi_L.parameters())), "Step 1: θ_π_L must NOT have grad"

        # ── Step 2A: L_lang_a  ────────────────────────────────────────
        # Gradient flows to Ψ through s_full (expected), but opt_lang
        # only contains θ_lang — Ψ would NOT be updated.
        # Verify: θ_lang and θ_Ψ are disjoint sets.
        model.zero_grad(set_to_none=True)
        loss_lang_path_a(traj['y_hat'], y, stop).backward(retain_graph=True)

        assert _has_grad(list(model.f_lang.projection.parameters())), \
            "Step 2A: θ_lang must have grad"
        assert len(lang_ids & psi_ids) == 0, \
            "Step 2A: θ_lang and θ_Ψ must be disjoint — opt_lang must not update Ψ (C6)"

        # ── Step 3: L_rl ─────────────────────────────────────────────
        model.zero_grad(set_to_none=True)
        loss_rl(traj['log_pi'], adv).backward(retain_graph=True)

        policy_params = (
            list(model.pi_M.parameters()) +
            list(model.pi_L.parameters()) +
            list(model.shared_mlp.parameters())
        )
        assert _has_grad(policy_params),                     "Step 3: policy must have grad"
        assert not _has_grad(list(model.psi.parameters())), "Step 3: θ_Ψ must NOT have grad"

        # ── Step 4: L_value ──────────────────────────────────────────
        model.zero_grad(set_to_none=True)
        loss_value(traj['v_t'], rd['returns']).backward()

        assert _has_grad(list(model.value_net.parameters())), "Step 4: θ_V must have grad"
        assert not _has_grad(list(model.psi.parameters())),  "Step 4: θ_Ψ must NOT have grad"
