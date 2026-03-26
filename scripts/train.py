"""
scripts/train.py

Main training entry point for MM-AdaptiveNN.

Usage:
  # Vision-only (ℳ={1}), ImageNet, main config
  python scripts/train.py --modalities vision --dataset imagenet \
      --data_root /data/imagenet --output_dir outputs/run_01

  # Audio-only (ℳ={2}), ESC-50
  python scripts/train.py --modalities audio --dataset esc50 \
      --data_root /data/esc50 --output_dir outputs/run_02

  # Joint visual-auditory (ℳ={1,2}), DAVE
  python scripts/train.py --modalities audiovisual --dataset dave \
      --data_root /data/dave --output_dir outputs/run_03

  # Ablation A7: Transformer state updater
  python scripts/train.py --modalities audiovisual --dataset imagenet \
      --psi transformer --output_dir outputs/ablation_a7
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import logging
import torch
import random
import numpy as np

from configs.default import get_config
from models.mm_adaptivenn import MMAdaptiveNN
from training.trainer import Trainer
from utils.grad_checks import run_all_checks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()

    # ── Dataset ──
    p.add_argument('--dataset', required=True,
                   choices=['imagenet', 'cub200', 'fgvc_aircraft',
                            'audioset', 'esc50',
                            'dave', 'calvin', 'avsbench', 'salicon'])
    p.add_argument('--data_root', required=True)
    p.add_argument('--modalities', default='audiovisual',
                   choices=['vision', 'audio', 'audiovisual'])

    # ── Model ──
    p.add_argument('--vision_backbone', default='vit_s16',
                   choices=['vit_s16', 'resnet50'])
    p.add_argument('--audio_backbone', default='ast',
                   choices=['ast', 'vggish'])
    p.add_argument('--psi', default='gru',
                   choices=['gru', 'lstm', 'transformer'])
    p.add_argument('--lang_backbone', default='clip',
                   choices=['clip', 't5'])
    p.add_argument('--num_classes', type=int, default=None)
    p.add_argument('--T', type=int, default=4)
    p.add_argument('--stop_lambda', type=float, default=0.5)

    # ── Training ──
    p.add_argument('--n_steps', type=int, default=100_000)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', type=str, default=None)

    # ── Output ──
    p.add_argument('--output_dir', default='outputs/default_run')
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--eval_every', type=int, default=1000)
    p.add_argument('--save_every', type=int, default=5000)

    # ── Hardware ──
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num_workers', type=int, default=8)

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args):
    modality_map = {'vision': [1], 'audio': [2], 'audiovisual': [1, 2]}
    num_classes_map = {
        'imagenet': 1000, 'cub200': 200, 'fgvc_aircraft': 100,
        'audioset': 527, 'esc50': 50,
        'dave': 4, 'calvin': 34, 'avsbench': 2,
    }

    cfg = get_config(
        run_name=f"mm_adaptivenn_{args.vision_backbone}_{args.audio_backbone}_{args.psi}",
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    cfg.model.modalities        = modality_map[args.modalities]
    cfg.encoder.vision_backbone = args.vision_backbone
    cfg.encoder.audio_backbone  = args.audio_backbone
    cfg.model.psi_backbone      = args.psi
    cfg.language.backbone       = args.lang_backbone
    cfg.training.T              = args.T
    cfg.training.stop_lambda    = args.stop_lambda
    cfg.training.n_total_steps  = args.n_steps
    cfg.training.batch_size     = args.batch_size
    cfg.model.num_classes       = args.num_classes or num_classes_map.get(args.dataset, 1000)
    return cfg


def build_dataloaders(args, cfg):
    """Build train and val DataLoaders for the specified dataset."""
    # Import the appropriate dataset module
    if args.dataset in ('imagenet', 'cub200', 'fgvc_aircraft'):
        from data.imagenet import build_imagenet_loaders
        return build_imagenet_loaders(args.data_root, cfg,
                                      dataset_name=args.dataset)
    elif args.dataset in ('audioset', 'esc50'):
        from data.audioset import build_audio_loaders
        return build_audio_loaders(args.data_root, cfg,
                                   dataset_name=args.dataset)
    elif args.dataset in ('dave', 'calvin', 'avsbench'):
        from data.audiovisual import build_audiovisual_loaders
        return build_audiovisual_loaders(args.data_root, cfg,
                                         dataset_name=args.dataset)
    elif args.dataset == 'salicon':
        from data.salicon import build_salicon_loaders
        return build_salicon_loaders(args.data_root, cfg)
    else:
        raise ValueError(
            f"Unknown dataset: {args.dataset}. "
            "Supported: imagenet, cub200, fgvc_aircraft, audioset, esc50, "
            "dave, calvin, avsbench, salicon"
        )


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Config: {args}")
    cfg = build_config(args)

    # ── Build model ──────────────────────────────────────────────────
    model = MMAdaptiveNN(cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_trainable:,} trainable / {n_total:,} total parameters")

    # ── AC constraint verification (before training) ─────────────────
    logger.info("Running AC constraint checks...")
    B = 2
    device = torch.device(cfg.device)
    dummy_X = {}
    if 1 in cfg.model.modalities:
        dummy_X[1] = torch.randn(B, 3, 128, 128, device=device)
    if 2 in cfg.model.modalities:
        dummy_X[2] = torch.randn(B, 1, 128, 100, device=device)
    dummy_w = ["test instruction"] * B

    # Run a TRAINING-MODE forward pass (gradients enabled, greedy=False) so
    # that AC-2 is a real check: s_sg must be detached even when grad tracking
    # is active.  Running under no_grad would make the check trivially pass.
    model.train()
    dummy_z = model.encode_language(dummy_w)
    stop_probs = model._stop_distribution(cfg.training.T,
                                           cfg.training.stop_lambda,
                                           device=device)
    dummy_traj = model.perception_loop(dummy_X, dummy_z,
                                       cfg.training.T, stop_probs,
                                       greedy=False)
    # Discard accumulated gradients from the diagnostic pass before training.
    model.zero_grad(set_to_none=True)

    ok = run_all_checks(model, dummy_X, dummy_w, traj=dummy_traj)
    if not ok:
        logger.error("AC constraint checks FAILED — fix before training")
        sys.exit(1)

    # ── Build dataloaders ─────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(args, cfg)

    # ── Build trainer ─────────────────────────────────────────────────
    trainer = Trainer(model, cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from {args.resume}")

    # ── Train ─────────────────────────────────────────────────────────
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader)
    logger.info("Training complete.")
    trainer.save_checkpoint(os.path.join(args.output_dir, 'final.pt'))


if __name__ == '__main__':
    main()
