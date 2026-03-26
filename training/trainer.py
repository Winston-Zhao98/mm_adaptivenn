"""
training/trainer.py

Four-step training algorithm implementing Theorem 1's gradient decomposition.

Each step updates a disjoint parameter group:
  Step 1: L_rep    → {θ_rep, θ_Ψ, θ_q}            full BPTT
  Step 2: L_lang   → θ_lang                         Path A + Path B
  Step 3: L_rl     → {θ_π^M, θ_π^L, θ_shared}      REINFORCE + reparam
  Step 4: L_value  → θ_V                            MSE

CRITICAL: Steps 1 and 2A use s_full (no detach). Steps 2B and 3 use s_sg.
This is enforced by the dual-graph perception loop in mm_adaptivenn.py.

Mixed Precision (AMP):
  Enabled by default when CUDA is available (cfg.training.use_amp=True).
  Uses a single GradScaler shared across all four optimisers.
  Each step follows the pattern: scale → backward → unscale → clip → step.
  scaler.update() is called once at the end of every train_step.

Reference: phase_b1_training.docx §1–§4, theorem_final.docx §4.
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.cuda.amp import GradScaler
from typing import Dict, Optional, List
import logging

from models.mm_adaptivenn import MMAdaptiveNN
from training.losses import (
    loss_rep, loss_lang_path_a, loss_lang_path_b,
    loss_rl, loss_value, loss_align,
    compute_rewards, compute_advantages,
)
from training.curriculum import CurriculumScheduler

logger = logging.getLogger(__name__)


def _try_tensorboard(log_dir: str):
    """Return a SummaryWriter if tensorboard is available, else None."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logging → {log_dir}")
        return writer
    except ImportError:
        logger.warning(
            "tensorboard not installed — install with: pip install tensorboard\n"
            "Falling back to console-only logging."
        )
        return None


class Trainer:

    def __init__(self, model: MMAdaptiveNN, cfg):
        self.model = model
        self.cfg = cfg
        self.curriculum = CurriculumScheduler(cfg)
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.global_step = 0

        # ── Mixed Precision ───────────────────────────────────────────────
        # AMP is only meaningful on CUDA. On CPU it degrades to FP32 silently.
        self.use_amp = (
            getattr(cfg.training, 'use_amp', True)
            and self.device.type == 'cuda'
        )
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info("Mixed precision (AMP / FP16) enabled.")
        else:
            logger.info("Mixed precision disabled (CPU or use_amp=False).")

        # ── TensorBoard writer ────────────────────────────────────────────
        tb_dir = os.path.join(cfg.output_dir, 'tensorboard')
        self.writer = _try_tensorboard(tb_dir)

        # ── Four separate optimisers (AC-3: independent lr for Path A vs B) ─
        self.opt_rep = AdamW(
            model.get_param_group('theta_rep') +
            model.get_param_group('theta_psi') +
            model.get_param_group('theta_q'),
            lr=cfg.training.lr_rep,
            weight_decay=cfg.training.weight_decay,
        )
        self.opt_lang = Adam(
            model.get_param_group('theta_lang'),
            lr=cfg.training.lr_lang_a,
        )
        self.opt_policy = Adam(
            model.get_param_group('theta_pi_M') +
            model.get_param_group('theta_pi_L') +
            model.get_param_group('theta_shared'),
            lr=cfg.training.lr_rl,
        )
        self.opt_value = Adam(
            model.get_param_group('theta_V'),
            lr=cfg.training.lr_value,
        )

        # Cosine LR scheduler for representation optimiser
        self.sched_rep = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_rep, T_max=cfg.training.n_total_steps
        )

        # Stopping distribution (fixed shape, annealed λ for efficiency curve)
        self._stop_probs = MMAdaptiveNN._stop_distribution(
            cfg.training.T, cfg.training.stop_lambda, device=self.device
        )

    # ── Main training step ────────────────────────────────────────────────────

    def train_step(
        self,
        X: Dict[int, torch.Tensor],
        y: torch.Tensor,
        w: list,
    ) -> Dict[str, float]:
        """
        One mini-batch iteration implementing Algorithm 1 from phase_b1_training.docx.
        Returns dict of scalar metrics for logging.

        AMP pattern per step:
          scaler.scale(loss).backward(...)
          scaler.unscale_(optimizer)       # unscale grads before clipping
          clip_grad_norm_(params, ...)
          scaler.step(optimizer)           # skips update if grads contain inf/nan
        scaler.update() is called ONCE at the very end of the step.
        """
        self.model.train()
        cfg = self.cfg
        T   = cfg.training.T

        X = {m: v.to(self.device) for m, v in X.items()}
        y = y.to(self.device)

        # ── Forward pass under autocast ───────────────────────────────────
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # AC-5: z computed once before loop
            z = self.model.encode_language(w)                  # (B, d_z)

            # Dual-graph perception loop
            traj = self.model.perception_loop(X, z, T, self._stop_probs)

            # Rewards, returns, advantages
            reward_dict = compute_rewards(
                traj['y_hat'], y, self._stop_probs, gamma=cfg.training.gamma
            )
            returns    = reward_dict['returns']                # (B, T)
            values     = traj['v_t']                           # (B, T)
            advantages = compute_advantages(
                returns, values,
                normalise=cfg.training.normalise_advantages,
            )

            # Pre-compute losses that share the s_full graph
            l_rep    = loss_rep(traj['y_hat'], y, self._stop_probs)
            l_lang_a = loss_lang_path_a(traj['y_hat'], y, self._stop_probs)

        # ── Step 1: Representation update (full BPTT) ─────────────────────
        self.opt_rep.zero_grad()
        self.scaler.scale(l_rep).backward(retain_graph=True)
        self.scaler.unscale_(self.opt_rep)
        nn.utils.clip_grad_norm_(
            self.model.get_param_group('theta_rep') +
            self.model.get_param_group('theta_psi') +
            self.model.get_param_group('theta_q'),
            max_norm=cfg.training.grad_clip_rep,
        )
        self.scaler.step(self.opt_rep)
        self.sched_rep.step()

        # ── Step 2A: Language Path A ───────────────────────────────────────
        self.opt_lang.zero_grad()
        self.scaler.scale(l_lang_a).backward(
            retain_graph=self.curriculum.rl_active
        )

        # ── Step 2B: Language Path B (Phase II+) ──────────────────────────
        l_lang_b_val = 0.
        if self.curriculum.rl_active:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                l_lang_b = loss_lang_path_b(traj['log_pi'], advantages)
            alpha_b_scale = self.curriculum.alpha_b_multiplier
            self.scaler.scale(l_lang_b * alpha_b_scale).backward(
                retain_graph=True
            )
            l_lang_b_val = l_lang_b.item()

        self.scaler.unscale_(self.opt_lang)
        nn.utils.clip_grad_norm_(
            self.model.get_param_group('theta_lang'),
            max_norm=cfg.training.grad_clip_lang_b,
        )
        self.scaler.step(self.opt_lang)

        # ── Step 3: Policy update (Phase II+) ─────────────────────────────
        l_rl_val = 0.
        if self.curriculum.rl_active:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pi_L_entropy = self.model.pi_L.entropy(
                    self.model.shared_mlp(
                        traj['s_sg'].reshape(-1, self.cfg.model.d_s).detach(),
                        z.unsqueeze(1).expand(-1, T, -1).reshape(
                            -1, self.cfg.language.d_z).detach(),
                    )
                ).reshape(-1, T)

                l_rl = loss_rl(
                    log_pi=traj['log_pi'],
                    advantages=advantages,
                    entropy=pi_L_entropy,
                    entropy_coef=self.curriculum.entropy_coef,
                )

            self.opt_policy.zero_grad()
            self.scaler.scale(l_rl).backward(retain_graph=True)
            self.scaler.unscale_(self.opt_policy)
            nn.utils.clip_grad_norm_(
                self.model.get_param_group('theta_pi_M') +
                self.model.get_param_group('theta_pi_L') +
                self.model.get_param_group('theta_shared'),
                max_norm=cfg.training.grad_clip_rl,
            )
            self.scaler.step(self.opt_policy)
            l_rl_val = l_rl.item()

        # ── Step 4: Value network ──────────────────────────────────────────
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            l_val = loss_value(values, returns)

        self.opt_value.zero_grad()
        self.scaler.scale(l_val).backward()
        self.scaler.unscale_(self.opt_value)
        nn.utils.clip_grad_norm_(
            self.model.get_param_group('theta_V'),
            max_norm=cfg.training.grad_clip_value,
        )
        self.scaler.step(self.opt_value)

        # ── Auxiliary: L_align (ℳ={1,2} only) ────────────────────────────
        l_align_val = 0.
        if 1 in cfg.model.modalities and 2 in cfg.model.modalities:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                o_v = self.model.f_rep['1'](X[1], l=None, glimpse=True)
                o_a = self.model.f_rep['2'](X[2], l=None, glimpse=True)
                l_align = loss_align(o_v, o_a)
            self.opt_rep.zero_grad(set_to_none=True)
            self.scaler.scale(cfg.training.lambda_align * l_align).backward()
            self.scaler.unscale_(self.opt_rep)
            nn.utils.clip_grad_norm_(
                self.model.get_param_group('theta_rep'),
                max_norm=cfg.training.grad_clip_rep,
            )
            self.scaler.step(self.opt_rep)
            l_align_val = l_align.item()

        # ── Update scaler (once per training step) ────────────────────────
        self.scaler.update()

        # ── Advance curriculum ────────────────────────────────────────────
        self.curriculum.advance()
        self.model.pi_L.sigma_max = self.curriculum.sigma_target

        # ── Training-time efficiency metric: E[t_o] ───────────────────────
        exp_steps = float(sum(
            (t + 1) * self._stop_probs[t].item()
            for t in range(T)
        ))

        metrics = {
            'loss/rep':             l_rep.item(),
            'loss/lang_a':          l_lang_a.item(),
            'loss/lang_b':          l_lang_b_val,
            'loss/rl':              l_rl_val,
            'loss/value':           l_val.item(),
            'loss/align':           l_align_val,
            'curriculum/phase':     self.curriculum.phase,
            'curriculum/sigma':     self.curriculum.sigma_target,
            'curriculum/alpha_b':   self.curriculum.alpha_b_multiplier,
            'efficiency/exp_steps': exp_steps,
            'lr/rep':               self.sched_rep.get_last_lr()[0],
            'amp/scale':            self.scaler.get_scale(),
        }

        # ── Write to TensorBoard ──────────────────────────────────────────
        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, self.global_step)

        self.global_step += 1
        return metrics

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, val_loader, max_steps: int = 500) -> Dict[str, float]:
        """
        Evaluate on val_loader. Returns:
          acc       : Top-1 accuracy
          acc_top5  : Top-5 accuracy
          exp_steps : Expected perception steps E[t_o] (efficiency metric)
          eval/modality_vision : fraction of steps selecting vision
          eval/modality_audio  : fraction of steps selecting audio (if ℳ={1,2})
        """
        self.model.eval()
        T = self.cfg.training.T
        stop_probs = MMAdaptiveNN._stop_distribution(
            T, self.cfg.training.stop_lambda, self.device
        )

        correct_1 = 0
        correct_5 = 0
        total = 0
        modality_counts: Dict[int, int] = {}
        total_decisions = 0

        for i, batch in enumerate(val_loader):
            if i >= max_steps:
                break
            X, y, w = self._unpack_batch(batch)
            X = {m: v.to(self.device) for m, v in X.items()}
            y_dev = y.to(self.device)
            B = y_dev.shape[0]

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                z    = self.model.encode_language(w)
                traj = self.model.perception_loop(X, z, T, stop_probs, greedy=True)

            # ── Top-1 and Top-5 accuracy ──────────────────────────────────
            logits = traj['y_hat'][:, -1, :].float()           # cast to FP32 for topk
            _, pred_top5 = logits.topk(min(5, logits.size(-1)), dim=-1)

            correct_1 += (pred_top5[:, 0] == y_dev).sum().item()
            correct_5 += (pred_top5 == y_dev.unsqueeze(1)).any(dim=1).sum().item()
            total     += B

            # ── Modality selection statistics ─────────────────────────────
            if 'm_ids' in traj:
                for m_id in traj['m_ids'].reshape(-1).tolist():
                    modality_counts[int(m_id)] = modality_counts.get(int(m_id), 0) + 1
                total_decisions += B * T

        # ── Expected steps (efficiency) ───────────────────────────────────
        exp_steps = float(sum(
            (t + 1) * stop_probs[t].item() for t in range(T)
        ))

        metrics: Dict[str, float] = {
            'acc':       correct_1 / max(total, 1),
            'acc_top5':  correct_5 / max(total, 1),
            'exp_steps': exp_steps,
        }

        if total_decisions > 0:
            for m_id, cnt in modality_counts.items():
                key = 'eval/modality_vision' if m_id == 1 else 'eval/modality_audio'
                metrics[key] = cnt / total_decisions

        # ── Write to TensorBoard ──────────────────────────────────────────
        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, self.global_step)

        return metrics

    # ── Full training loop ─────────────────────────────────────────────────────

    def train(self, train_loader, val_loader=None):
        cfg = self.cfg
        step = 0
        for batch in train_loader:
            if step >= cfg.training.n_total_steps:
                break
            X, y, w = self._unpack_batch(batch)
            metrics = self.train_step(X, y, w)

            if step % cfg.log_every == 0:
                loss_summary = {k: f"{v:.4f}" for k, v in metrics.items()
                                if k.startswith('loss/')}
                logger.info(
                    f"Step {step:6d} | "
                    f"phase={int(metrics['curriculum/phase'])} | "
                    f"E[t]={metrics['efficiency/exp_steps']:.2f} | "
                    f"scale={metrics['amp/scale']:.0f} | "
                    f"{loss_summary}"
                )

            if val_loader and step % cfg.eval_every == 0 and step > 0:
                val_metrics = self.evaluate(val_loader)
                logger.info(
                    f"Step {step:6d} | Val "
                    f"Top-1={val_metrics['acc']:.4f} "
                    f"Top-5={val_metrics['acc_top5']:.4f} "
                    f"E[t]={val_metrics['exp_steps']:.2f}"
                )

            if step % cfg.save_every == 0 and step > 0:
                self.save_checkpoint(f"{cfg.output_dir}/ckpt_{step:06d}.pt")

            step += 1

        if self.writer is not None:
            self.writer.close()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            return batch[0], batch[1], batch[2]
        return batch['X'], batch['y'], batch['w']

    def save_checkpoint(self, path: str):
        import dataclasses
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model':       self.model.state_dict(),
            'opt_rep':     self.opt_rep.state_dict(),
            'opt_lang':    self.opt_lang.state_dict(),
            'opt_policy':  self.opt_policy.state_dict(),
            'opt_value':   self.opt_value.state_dict(),
            'curriculum':  self.curriculum.state_dict(),
            'scaler':      self.scaler.state_dict(),
            'global_step': self.global_step,
            'cfg':         dataclasses.asdict(self.cfg),
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.opt_rep.load_state_dict(ckpt['opt_rep'])
        self.opt_lang.load_state_dict(ckpt['opt_lang'])
        self.opt_policy.load_state_dict(ckpt['opt_policy'])
        self.opt_value.load_state_dict(ckpt['opt_value'])
        self.curriculum.load_state_dict(ckpt['curriculum'])
        if 'scaler' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler'])
        if 'global_step' in ckpt:
            self.global_step = ckpt['global_step']
        logger.info(f"Loaded checkpoint: {path}  (step {self.global_step})")
