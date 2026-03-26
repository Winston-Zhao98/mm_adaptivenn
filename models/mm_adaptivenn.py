"""
models/mm_adaptivenn.py

Full MM-AdaptiveNN model.

The perception loop maintains TWO state chains:
  s_full: full computation graph — gradients flow through Ψ (for Step 1, 2A)
  s_sg:   stop-gradient chain   — s_{t-1} detached at each step (for Steps 2B, 3)

This dual-graph design is the central implementation challenge (AC-2 / C4).
See trainer.py for how the two chains are used in the four-step update.

Trajectory dict returned by perception_loop() contains all quantities needed
for gradient computation in all four training steps.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .language_encoder import build_language_encoder
from .modality_encoders import build_modality_encoders
from .state_updater import build_state_updater
from .policy_networks import SharedMLP, ModalitySelectionPolicy, LocationAttentionPolicy, TaskHead, ValueNetwork


class MMAdaptiveNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.modalities = cfg.model.modalities

        # ── Module construction ─────────────────────────────────────
        self.f_lang   = build_language_encoder(cfg)
        self.f_rep    = build_modality_encoders(cfg)
        self.psi      = build_state_updater(cfg)
        self.shared_mlp = SharedMLP(
            d_s=cfg.model.d_s, d_z=cfg.language.d_z,
            d_h=cfg.model.d_h, n_layers=cfg.model.shared_mlp_layers,
        )
        self.pi_M = ModalitySelectionPolicy(
            d_h=cfg.model.d_h, modalities=cfg.model.modalities,
        )
        self.pi_L = LocationAttentionPolicy(
            d_h=cfg.model.d_h, sigma_max=cfg.model.sigma_max,
        )
        self.task_head = TaskHead(
            d_s=cfg.model.d_s, d_z=cfg.language.d_z,
            num_classes=cfg.model.num_classes,
            fusion=cfg.model.task_head_fusion,
        )
        self.value_net = ValueNetwork(d_s=cfg.model.d_s)

        # ── Parameter group bookkeeping ─────────────────────────────
        # Used in trainer.py to assign parameters to the correct optimiser.
        self._param_groups = self._build_param_groups()

    # ── Parameter groups (7 groups from Theorem 1) ──────────────────────────

    def _build_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """
        Returns dict of parameter groups matching Theorem 1 / arch_03 §5.
        θ_π^M ∩ θ_π^L = ∅ is guaranteed by using separate nn.Modules.
        """
        groups = {
            'theta_rep': [],          # {θ_rep^(m)} — modality encoders + projections
            'theta_psi': list(self.psi.parameters()),
            'theta_q':   list(self.task_head.parameters()),
            'theta_lang': list(self.f_lang.projection.parameters()),  # trainable part only
            'theta_pi_M': list(self.pi_M.parameters()),
            'theta_pi_L': list(self.pi_L.parameters()),
            'theta_shared': list(self.shared_mlp.parameters()),
            'theta_V':   list(self.value_net.parameters()),
        }
        for enc in self.f_rep.values():
            # Only trainable parameters (unfrozen layers + projection head)
            groups['theta_rep'].extend([p for p in enc.parameters() if p.requires_grad])
        return groups

    def get_param_group(self, name: str) -> List[nn.Parameter]:
        return self._param_groups[name]

    # ── Pre-computation (AC-5) ──────────────────────────────────────────────

    def encode_language(self, w: list) -> torch.Tensor:
        """
        Compute z = f_lang(w) ONCE per sample, before the perception loop.
        AC-5: z is fixed for the entire loop.
        Returns z: (B, d_z)
        """
        return self.f_lang(w)

    # ── Perception loop ─────────────────────────────────────────────────────

    def perception_loop(
        self,
        X: Dict[int, torch.Tensor],    # {modality_id: tensor}
        z: torch.Tensor,               # (B, d_z), pre-computed, detached or not
        T: int,
        stop_probs: torch.Tensor,      # (T,) P(t_o = t) for each step
        greedy: bool = False,
    ) -> Dict:
        """
        Run T-step perception loop.

        Maintains DUAL state chains:
          s_full: for Step 1 (L_rep) — full BPTT, s_{t-1} NOT detached in Ψ
          s_sg:   for Steps 2B, 3    — s_{t-1} detached at each step

        Returns trajectory dict containing all quantities for gradient computation.

        AC-1 check: f_rep^(m) is called with (X[m], l_t) only — z never enters.
        AC-2 check: shared_mlp receives s_sg (detached), not s_full.
        """
        B = z.shape[0]
        device = z.device

        # Initialise dual state chains
        s_full = self.psi.initial_state(B, device)         # full graph
        s_sg   = s_full.detach().clone()                   # stop-gradient chain

        # ── Global glimpse (t=0): initialise states ──────────────────
        with torch.no_grad() if greedy else torch.enable_grad():
            # Global glimpse uses full graph too (contributes to L_rep)
            o0_list = []
            for m_id in self.modalities:
                enc = self.f_rep[str(m_id)]
                o0 = enc(X[m_id], l=None, glimpse=True)    # AC-1: no z
                o0_list.append(o0)
            # Average global glimpses across modalities
            o0 = torch.stack(o0_list, dim=0).mean(0)      # (B, d)

        s_full = self.psi(s_full, o0)                      # full BPTT
        s_sg   = self.psi(s_sg.detach(), o0.detach()).detach()

        # ── Trajectory storage ───────────────────────────────────────
        traj = {
            's_full': [],        # s_t from full graph (Step 1)
            's_sg': [],          # s_{t-1} detached (Steps 2B, 3)
            'o_t': [],           # observations
            'm_idx': [],         # modality index (into self.modalities list)
            'm_ids': [],         # actual modality IDs
            'l_t': [],           # attention locations
            'log_pi_M': [],      # log p(m_t | sg(s), z)
            'log_pi_L': [],      # log p(l_t | m_t, sg(s), z)
            'log_pi': [],        # log p(a_t) = log_pi_M + log_pi_L
            'y_hat': [],         # task predictions ŷ_t (full graph)
            'r_t': [],           # placeholder (filled in trainer with loss)
            'v_t': [],           # V^π(sg(s_{t-1}))
            'stop_probs': stop_probs,
        }
        # History buffer for CausalTransformer Ψ (stores all o_t seen so far)
        obs_history: list = []

        for t in range(T):
            # ── Policy: uses sg(s_{t-1}) (AC-2) ─────────────────────
            h = self.shared_mlp(s_sg, z)                   # z-conditioned, sg(s)

            # π^M: modality selection (REINFORCE)
            m_idx, log_pi_M = self.pi_M.select(h, greedy=greedy)
            m_ids = self.pi_M.modality_ids(m_idx)          # actual IDs

            # π^L: location attention (reparameterisation)
            l_t, log_pi_L = self.pi_L.select(h, m_ids)

            # Value estimate from sg(s)
            v_t = self.value_net(s_sg)

            # ── Perception: AC-1 enforced by encoder interface ───────
            o_t = self._encode_observation(X, m_ids, l_t)  # no z in here

            # ── Dual state update ────────────────────────────────────
            obs_history.append(o_t)
            history_tensor = torch.stack(obs_history, dim=1)          # (B, t+1, d)
            history_tensor_sg = history_tensor.detach()

            # Full graph: s_{t-1} NOT detached → BPTT flows through Ψ (Step 1)
            if hasattr(self.psi, 'max_seq_len'):  # CausalTransformer
                s_full_new = self.psi(s_full, o_t,
                                      history=history_tensor, step=t)
            else:
                s_full_new = self.psi(s_full, o_t)

            # SG chain: s_{t-1} detached → no grad through state (Steps 2B, 3)
            if hasattr(self.psi, 'max_seq_len'):
                s_sg_new = self.psi(s_sg.detach(), o_t.detach(),
                                    history=history_tensor_sg, step=t).detach()
            else:
                s_sg_new = self.psi(s_sg.detach(), o_t.detach()).detach()

            # ── Task prediction: uses full graph s_t ─────────────────
            y_hat = self.task_head(s_full_new, z)           # s_full + z (AC-1)

            # Store
            traj['s_full'].append(s_full_new)
            traj['s_sg'].append(s_sg)                       # s_{t-1} for policy inputs
            traj['o_t'].append(o_t)
            traj['m_idx'].append(m_idx)
            traj['m_ids'].append(m_ids)
            traj['l_t'].append(l_t)
            traj['log_pi_M'].append(log_pi_M)
            traj['log_pi_L'].append(log_pi_L)
            traj['log_pi'].append(log_pi_M + log_pi_L)
            traj['y_hat'].append(y_hat)
            traj['v_t'].append(v_t)

            # Advance states
            s_full = s_full_new
            s_sg   = s_sg_new

        # Stack across time
        for key in ['s_full', 's_sg', 'o_t', 'm_idx', 'm_ids', 'l_t',
                    'log_pi_M', 'log_pi_L', 'log_pi', 'y_hat', 'v_t']:
            traj[key] = torch.stack(traj[key], dim=1)       # (B, T, ...)

        return traj

    def _encode_observation(
        self,
        X: Dict[int, torch.Tensor],
        m_ids: torch.Tensor,
        l_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode one observation per sample using the selected modality.
        AC-1: no z here — only (X[m], l_t).
        m_ids: (B,) actual modality IDs
        l_t:   (B, 2) location (x,y) for vision; (B, 2)[0] = τ for audio
        """
        B = m_ids.shape[0]
        o_t = torch.zeros(B, self.cfg.model.d, device=m_ids.device)

        for m in self.modalities:
            mask = (m_ids == m)
            if not mask.any():
                continue
            enc = self.f_rep[str(m)]
            l_m = l_t[mask]                                # (B_m, 2)
            if m == 1:
                loc = l_m                                  # (x, y)
            else:
                loc = l_m[:, :1]                           # τ only
            o_m = enc(X[m][mask], l=loc, glimpse=False)   # AC-1 enforced
            o_t[mask] = o_m

        return o_t

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        X: Dict[int, torch.Tensor],
        w: list,
        T: int = 4,
        stop_lambda: float = 0.5,
    ) -> torch.Tensor:
        """
        Greedy inference. Returns class logits at final step.
        AC-5: z computed once before loop.
        """
        B = next(iter(X.values())).shape[0]
        z = self.encode_language(w)
        stop_probs = self._stop_distribution(T, stop_lambda,
                                             device=z.device)
        traj = self.perception_loop(X, z, T, stop_probs, greedy=True)
        return traj['y_hat'][:, -1, :]                     # last step prediction

    @staticmethod
    def _stop_distribution(T: int, lam: float,
                           device: torch.device) -> torch.Tensor:
        """Exponential stopping: P(t_o=t) ∝ (1-λ)^{t-1} λ, normalised."""
        probs = torch.tensor(
            [(1 - lam) ** (t - 1) * lam for t in range(1, T + 1)],
            device=device, dtype=torch.float32,
        )
        return probs / probs.sum()
