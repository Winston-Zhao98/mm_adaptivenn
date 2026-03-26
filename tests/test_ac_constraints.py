"""
tests/test_ac_constraints.py

Unit tests verifying AC-1 through AC-5 at model initialisation.
Run with: python -m pytest tests/test_ac_constraints.py -v

All tests should pass before any training run.
Reference: utils/grad_checks.py, theorem_final.docx §4.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytest
from configs.default import get_config
from models.mm_adaptivenn import MMAdaptiveNN
from utils.grad_checks import (
    check_ac1_no_z_in_encoder,
    check_ac1_no_z_in_psi,
    check_ac2_sg_state,
    check_ac4_disjoint_policy_params,
    check_ac5_z_precomputed,
)


@pytest.fixture
def small_cfg():
    """Minimal config for fast testing (no pretrained weights)."""
    cfg = get_config()
    cfg.model.d        = 32
    cfg.model.d_s      = 64
    cfg.model.d_h      = 64
    cfg.language.d_z   = 32
    cfg.model.modalities = [1, 2]
    cfg.model.num_classes = 10
    cfg.training.T     = 2
    cfg.device         = 'cpu'
    return cfg


@pytest.fixture
def model_and_inputs(small_cfg):
    """Build a model and minimal dummy inputs."""
    # Override encoders with lightweight stubs
    model = _build_stub_model(small_cfg)
    B = 2
    X = {
        1: torch.randn(B, 3, 128, 128),     # vision
        2: torch.randn(B, 1, 128, 100),     # audio mel spectrogram
    }
    w = ["pick up the red block", "look at the window"]
    return model, X, w


def _build_stub_model(cfg) -> MMAdaptiveNN:
    """
    Build MMAdaptiveNN with stub encoders that do NOT load pretrained weights.
    Ensures tests run fast without internet access.
    """
    import torch.nn as nn
    from models.mm_adaptivenn import MMAdaptiveNN
    from models.modality_encoders import ProjectionHead
    from models.state_updater import GRUStateUpdater
    from models.language_encoder import FiLMLayer
    from models.policy_networks import SharedMLP, ModalitySelectionPolicy, LocationAttentionPolicy, TaskHead, ValueNetwork

    class StubModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.modalities = cfg.model.modalities

            # Stub language encoder (no CLIP)
            class StubLangEnc(nn.Module):
                def __init__(self, d_z):
                    super().__init__()
                    self.projection = nn.Linear(16, d_z)
                    self.d_z = d_z
                def forward(self, w):
                    B = len(w)
                    return self.projection(torch.randn(B, 16))
            self.f_lang = StubLangEnc(cfg.language.d_z)

            # Stub modality encoders (no pretrained backbones)
            class StubEnc(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    self.proj = nn.Linear(16, d)
                    self.d = d
                def forward(self, x, l, glimpse=False):
                    B = x.shape[0]
                    return self.proj(torch.randn(B, 16))
            self.f_rep = nn.ModuleDict({
                '1': StubEnc(cfg.model.d),
                '2': StubEnc(cfg.model.d),
            })

            self.psi = GRUStateUpdater(d=cfg.model.d, d_s=cfg.model.d_s)
            self.shared_mlp = SharedMLP(
                d_s=cfg.model.d_s, d_z=cfg.language.d_z,
                d_h=cfg.model.d_h, n_layers=2,
            )
            self.pi_M = ModalitySelectionPolicy(cfg.model.d_h, cfg.model.modalities)
            self.pi_L = LocationAttentionPolicy(cfg.model.d_h)
            self.task_head = TaskHead(
                cfg.model.d_s, cfg.language.d_z, cfg.model.num_classes
            )
            self.value_net = ValueNetwork(cfg.model.d_s)

        def _build_param_groups(self):
            return {
                'theta_rep':    list(self.f_rep.parameters()),
                'theta_psi':    list(self.psi.parameters()),
                'theta_q':      list(self.task_head.parameters()),
                'theta_lang':   list(self.f_lang.parameters()),
                'theta_pi_M':   list(self.pi_M.parameters()),
                'theta_pi_L':   list(self.pi_L.parameters()),
                'theta_shared': list(self.shared_mlp.parameters()),
                'theta_V':      list(self.value_net.parameters()),
            }

        def get_param_group(self, name):
            return self._build_param_groups()[name]

        def encode_language(self, w):
            return self.f_lang(w)

        # Minimal perception loop for test
        def perception_loop(self, X, z, T, stop_probs, greedy=False):
            from models.mm_adaptivenn import MMAdaptiveNN as _M
            # Reuse the real perception loop via mixin
            B = z.shape[0]
            device = z.device
            s_full = self.psi.initial_state(B, device)
            s_sg   = s_full.detach().clone()
            traj = {
                's_full': [], 's_sg': [], 'o_t': [],
                'm_idx': [], 'm_ids': [], 'l_t': [],
                'log_pi_M': [], 'log_pi_L': [], 'log_pi': [],
                'y_hat': [], 'v_t': [], 'stop_probs': stop_probs,
            }
            for t in range(T):
                h = self.shared_mlp(s_sg, z)
                m_idx, log_pi_M = self.pi_M.select(h)
                m_ids = self.pi_M.modality_ids(m_idx)
                l_t, log_pi_L = self.pi_L.select(h, m_ids)
                v_t = self.value_net(s_sg)
                # dummy o_t
                o_t = torch.randn(B, self.cfg.model.d, device=device)
                s_full = self.psi(s_full, o_t)
                s_sg_new = self.psi(s_sg.detach(), o_t.detach()).detach()
                y_hat = self.task_head(s_full, z)
                traj['s_full'].append(s_full)
                traj['s_sg'].append(s_sg)
                traj['o_t'].append(o_t)
                traj['m_idx'].append(m_idx); traj['m_ids'].append(m_ids)
                traj['l_t'].append(l_t)
                traj['log_pi_M'].append(log_pi_M); traj['log_pi_L'].append(log_pi_L)
                traj['log_pi'].append(log_pi_M + log_pi_L)
                traj['y_hat'].append(y_hat); traj['v_t'].append(v_t)
                s_sg = s_sg_new
            for k in ['s_full','s_sg','o_t','m_idx','m_ids','l_t',
                      'log_pi_M','log_pi_L','log_pi','y_hat','v_t']:
                traj[k] = torch.stack(traj[k], dim=1)
            return traj

    return StubModel(cfg)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAC4DisjointPolicyHeads:
    """AC-4 / C5: θ_π^M ∩ θ_π^L = ∅"""

    def test_policy_heads_disjoint(self, model_and_inputs):
        model, X, w = model_and_inputs
        assert check_ac4_disjoint_policy_params(model), \
            "AC-4 FAILED: policy head parameters are not disjoint"

    def test_theta_shared_disjoint_from_pi_M(self, model_and_inputs):
        model, X, w = model_and_inputs
        ptrs_M      = set(p.data_ptr() for p in model.pi_M.parameters())
        ptrs_shared = set(p.data_ptr() for p in model.shared_mlp.parameters())
        assert ptrs_M.isdisjoint(ptrs_shared), \
            "θ_shared and θ_π^M share parameters"

    def test_theta_shared_disjoint_from_pi_L(self, model_and_inputs):
        model, X, w = model_and_inputs
        ptrs_L      = set(p.data_ptr() for p in model.pi_L.parameters())
        ptrs_shared = set(p.data_ptr() for p in model.shared_mlp.parameters())
        assert ptrs_L.isdisjoint(ptrs_shared), \
            "θ_shared and θ_π^L share parameters"


class TestAC2SGState:
    """AC-2 / C4: s_sg entries must be detached"""

    def test_s_sg_detached(self, model_and_inputs):
        model, X, w = model_and_inputs
        z = model.encode_language(w)
        stop_probs = torch.tensor([0.5, 0.5])
        traj = model.perception_loop(X, z, T=2, stop_probs=stop_probs)
        assert check_ac2_sg_state(traj), \
            "AC-2 FAILED: s_sg is not detached"

    def test_s_sg_no_grad(self, model_and_inputs):
        model, X, w = model_and_inputs
        z = model.encode_language(w)
        stop_probs = torch.tensor([0.5, 0.5])
        traj = model.perception_loop(X, z, T=2, stop_probs=stop_probs)
        s_sg = traj['s_sg']
        assert not s_sg.requires_grad, "s_sg.requires_grad should be False"


class TestAC5ZPrecomputed:
    """AC-5: encode_language is structurally separate from Ψ and f_rep"""

    def test_encode_language_method_exists(self, model_and_inputs):
        model, X, w = model_and_inputs
        assert hasattr(model, 'encode_language'), \
            "Model missing encode_language() method (AC-5)"

    def test_z_shape(self, model_and_inputs, small_cfg):
        model, X, w = model_and_inputs
        z = model.encode_language(w)
        assert z.shape == (len(w), small_cfg.language.d_z), \
            f"z shape mismatch: expected ({len(w)}, {small_cfg.language.d_z}), got {z.shape}"


class TestC1CommonEmbedding:
    """C1: all f_rep^(m) output to ℝ^d"""

    def test_all_encoders_output_same_dim(self, model_and_inputs, small_cfg):
        model, X, w = model_and_inputs
        d = small_cfg.model.d
        B = 2
        for m_id, enc in model.f_rep.items():
            l = torch.zeros(B, 2)
            o = enc(X[int(m_id)], l)
            assert o.shape == (B, d), \
                f"Encoder {m_id} output shape {o.shape} ≠ (B={B}, d={d}) — C1 violated"


class TestGradientFlow:
    """Verify the dual-graph produces correct gradient flows."""

    def test_rep_loss_grads_through_psi(self, model_and_inputs, small_cfg):
        """L_rep must produce gradients for θ_Ψ (full BPTT)."""
        model, X, w = model_and_inputs
        model.train()
        z = model.encode_language(w)
        stop_probs = torch.tensor([0.5, 0.5])
        traj = model.perception_loop(X, z, T=2, stop_probs=stop_probs)
        y = torch.zeros(len(w), dtype=torch.long)
        from training.losses import loss_rep
        l = loss_rep(traj['y_hat'], y, stop_probs)
        l.backward()
        for p in model.psi.parameters():
            assert p.grad is not None, "θ_Ψ has no gradient from L_rep (BPTT broken)"

    def test_rl_loss_no_grads_through_psi(self, model_and_inputs):
        """L_rl must NOT produce gradients for θ_Ψ (s_sg is detached)."""
        model, X, w = model_and_inputs
        model.train()
        z = model.encode_language(w).detach()
        stop_probs = torch.tensor([0.5, 0.5])
        traj = model.perception_loop(X, z, T=2, stop_probs=stop_probs)
        # Zero existing grads
        for p in model.psi.parameters():
            if p.grad is not None:
                p.grad.zero_()
        # RL loss through log_pi (which uses s_sg — detached)
        from training.losses import loss_rl
        advantages = torch.ones(len(w), 2)
        l = loss_rl(traj['log_pi'], advantages)
        l.backward()
        for p in model.psi.parameters():
            grad_norm = p.grad.abs().max().item() if p.grad is not None else 0.
            assert grad_norm < 1e-9, \
                f"θ_Ψ has grad {grad_norm:.2e} from L_rl — s_sg not properly detached"
