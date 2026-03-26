"""
scripts/evaluate.py

Evaluation script for MM-AdaptiveNN.

Supports:
  1. Standard classification accuracy (ImageNet, CUB-200, ESC-50, DAVE)
  2. Efficiency evaluation (GFLOPs vs accuracy curve, sweep λ)
  3. Human attention comparison (SALICON fixation prediction)
  4. AC constraint verification on a trained model

Usage:
  # Standard evaluation
  python scripts/evaluate.py --checkpoint outputs/run_01/final.pt \
      --dataset imagenet --data_root /data/imagenet

  # Efficiency sweep
  python scripts/evaluate.py --checkpoint outputs/run_01/final.pt \
      --dataset imagenet --data_root /data/imagenet \
      --eval_mode efficiency --lambda_sweep 0.2 0.3 0.5 0.7 0.9

  # SALICON human attention comparison
  python scripts/evaluate.py --checkpoint outputs/run_01/final.pt \
      --dataset salicon --data_root /data/salicon \
      --eval_mode attention
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import logging
import torch
import numpy as np
from typing import List

from configs.default import get_config
from models.mm_adaptivenn import MMAdaptiveNN
from utils.grad_checks import run_all_checks

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ── SALICON dataset ───────────────────────────────────────────────────────────

class SALICONDataset(torch.utils.data.Dataset):
    """
    SALICON: Saliency in Context (10K MS-COCO images + fixation density maps).
    Used to evaluate whether MM-AdaptiveNN's attention matches human gaze.

    Expected structure:
      data_root/
        images/{split}/{image_id}.jpg
        fixations/{split}/{image_id}.mat   (fixation density map)

    Download from: http://salicon.net/challenge-2017/
    """
    def __init__(self, data_root, split='val'):
        import glob
        from PIL import Image
        import scipy.io as sio
        self.Image = Image
        self.sio = sio
        self.data_root = data_root
        self.split = split

        img_dir = os.path.join(data_root, 'images', split)
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        if not self.img_paths:
            raise FileNotFoundError(
                f"No SALICON images found in {img_dir}\n"
                "Download from: http://salicon.net/challenge-2017/"
            )
        self.transform = __import__('torchvision').transforms.Compose([
            __import__('torchvision').transforms.Resize((224, 224)),
            __import__('torchvision').transforms.ToTensor(),
            __import__('torchvision').transforms.Normalize(
                [0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]

        img = self.Image.open(img_path).convert('RGB')
        img_t = self.transform(img)

        # Load fixation map
        fix_path = os.path.join(self.data_root, 'fixations', self.split, f'{img_id}.mat')
        if os.path.exists(fix_path):
            mat = self.sio.loadmat(fix_path)
            fixmap = torch.from_numpy(mat['fixation_map'].astype(np.float32))
            # Normalise to [0,1]
            if fixmap.max() > 0:
                fixmap = fixmap / fixmap.max()
        else:
            fixmap = torch.zeros(480, 640)

        return img_t, fixmap, img_id


def evaluate_attention(model, data_root: str, cfg, device):
    """
    Compare model's visual fixation sequence against SALICON human fixations.
    Metric: Normalised Scanpath Salience (NSS) — higher is better.
    """
    from torch.utils.data import DataLoader
    ds = SALICONDataset(data_root, 'val')
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

    nss_scores = []
    model.eval()

    for imgs, fixmaps, img_ids in loader:
        imgs = imgs.to(device)
        B = imgs.shape[0]
        X = {1: imgs}
        w = ["what is the most salient object in the scene?"] * B
        z = model.encode_language(w)
        stop_probs = MMAdaptiveNN._stop_distribution(cfg.training.T,
                                                      cfg.training.stop_lambda, device)

        with torch.no_grad():
            traj = model.perception_loop(X, z, cfg.training.T, stop_probs, greedy=True)

        # Collect fixation locations (vision modality only)
        l_seq = traj['l_t']                               # (B, T, 2)
        m_ids = traj['m_ids']                             # (B, T)

        for b in range(B):
            fixmap_b = fixmaps[b].numpy()                 # (H, W)
            H, W = fixmap_b.shape

            # Extract vision fixations
            nss_vals = []
            for t in range(cfg.training.T):
                if m_ids[b, t].item() == 1:               # vision step
                    x_norm = l_seq[b, t, 0].item()
                    y_norm = l_seq[b, t, 1].item()
                    px = int(x_norm * W)
                    py = int(y_norm * H)
                    px = max(0, min(px, W-1))
                    py = max(0, min(py, H-1))
                    # NSS at this fixation
                    mean_f = fixmap_b.mean()
                    std_f  = fixmap_b.std() + 1e-8
                    nss = (fixmap_b[py, px] - mean_f) / std_f
                    nss_vals.append(nss)

            if nss_vals:
                nss_scores.append(np.mean(nss_vals))

    mean_nss = float(np.mean(nss_scores)) if nss_scores else 0.
    logger.info(f"SALICON NSS: {mean_nss:.4f} (N={len(nss_scores)} images)")
    return {'nss': mean_nss}


def evaluate_efficiency(model, data_loader, cfg, device,
                        lambda_values: List[float] = None):
    """
    Sweep stopping λ to generate efficiency-accuracy trade-off curve.
    Returns list of (lambda, accuracy, avg_steps) tuples.
    """
    if lambda_values is None:
        lambda_values = [0.2, 0.3, 0.5, 0.7, 0.9]

    results = []
    model.eval()

    for lam in lambda_values:
        correct, total, total_steps = 0, 0, 0.0
        stop_probs = MMAdaptiveNN._stop_distribution(cfg.training.T, lam, device)
        # Expected steps = Σ_t t * P(t_o=t)
        exp_steps = sum((t+1) * stop_probs[t].item() for t in range(cfg.training.T))

        for batch in data_loader:
            X, y, w = batch
            X = {m: v.to(device) for m, v in X.items()}
            y = y.to(device)

            with torch.no_grad():
                z = model.encode_language(w)
                traj = model.perception_loop(X, z, cfg.training.T,
                                              stop_probs, greedy=True)

            logits = traj['y_hat'][:, -1, :]
            correct += (logits.argmax(-1) == y).sum().item()
            total   += y.shape[0]

        acc = correct / max(total, 1)
        results.append({'lambda': lam, 'acc': acc, 'exp_steps': exp_steps})
        logger.info(f"λ={lam:.1f}: acc={acc:.4f}, E[steps]={exp_steps:.2f}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--eval_mode', default='accuracy',
                   choices=['accuracy', 'efficiency', 'attention', 'ac_check'])
    p.add_argument('--lambda_sweep', nargs='+', type=float,
                   default=[0.2, 0.3, 0.5, 0.7, 0.9])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Reconstruct the exact config used at training time.
    # Checkpoints saved by Trainer.save_checkpoint() embed a 'cfg' dict.
    # Fall back to defaults only for legacy checkpoints that lack it.
    if 'cfg' in ckpt:
        import dataclasses
        from configs.default import Config, EncoderConfig, LanguageEncoderConfig, ModelConfig, TrainingConfig
        raw = ckpt['cfg']
        cfg = Config(
            encoder=EncoderConfig(**raw['encoder']),
            language=LanguageEncoderConfig(**raw['language']),
            model=ModelConfig(**raw['model']),
            training=TrainingConfig(**raw['training']),
            run_name=raw.get('run_name', 'mm_adaptivenn'),
            seed=raw.get('seed', 42),
            device=args.device,
            num_workers=raw.get('num_workers', 4),
            log_every=raw.get('log_every', 100),
            eval_every=raw.get('eval_every', 1000),
            save_every=raw.get('save_every', 5000),
            output_dir=raw.get('output_dir', 'outputs/'),
        )
        logger.info("Restored cfg from checkpoint.")
    else:
        cfg = get_config()
        logger.warning(
            "Checkpoint does not contain 'cfg' — using default config. "
            "Evaluation may be incorrect if training used non-default settings."
        )

    model = MMAdaptiveNN(cfg)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    if args.eval_mode == 'ac_check':
        B = 2
        dummy_X = {1: torch.randn(B,3,128,128,device=device)}
        if 2 in cfg.model.modalities:
            dummy_X[2] = torch.randn(B,1,128,100,device=device)
        run_all_checks(model, dummy_X, ["test"]*B)
        return

    if args.eval_mode == 'attention':
        evaluate_attention(model, args.data_root, cfg, device)
        return

    # Build dataloader
    cfg.training.batch_size = args.batch_size
    if args.dataset in ('imagenet', 'cub200', 'fgvc_aircraft'):
        from data.imagenet import build_imagenet_loaders
        _, val_loader = build_imagenet_loaders(args.data_root, cfg, args.dataset)
    elif args.dataset in ('audioset', 'esc50'):
        from data.audioset import build_audio_loaders
        _, val_loader = build_audio_loaders(args.data_root, cfg, args.dataset)
    else:
        from data.audiovisual import build_audiovisual_loaders
        _, val_loader = build_audiovisual_loaders(args.data_root, cfg, args.dataset)

    if args.eval_mode == 'efficiency':
        evaluate_efficiency(model, val_loader, cfg, device, args.lambda_sweep)
    else:
        from training.trainer import Trainer
        trainer = Trainer(model, cfg)
        metrics = trainer.evaluate(val_loader)
        logger.info(f"Accuracy: {metrics}")


if __name__ == '__main__':
    main()
