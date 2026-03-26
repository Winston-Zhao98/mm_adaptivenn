"""
configs/default.py
All hyperparameters for MM-AdaptiveNN.
Corresponds to the hyperparameter table in phase_b1_training.docx §7.
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class EncoderConfig:
    # ── Visual encoder ──────────────────────────────────────────────
    vision_backbone: Literal['vit_s16', 'resnet50'] = 'vit_s16'
    vision_pretrained: str = 'imagenet21k'          # or 'imagenet1k'
    vision_patch_size: int = 96                     # P in pixels
    vision_glimpse_size: int = 64                   # global-glimpse downsampled size
    vision_finetune_layers: int = 4                 # last N layers unfrozen (ViT)

    # ── Audio encoder ───────────────────────────────────────────────
    audio_backbone: Literal['ast', 'vggish'] = 'ast'
    audio_pretrained: str = 'audioset'
    audio_window_sec: float = 1.0                   # Δτ in seconds
    audio_glimpse_size: int = 128                   # global-glimpse downsampled size
    audio_finetune_layers: int = 3                  # last N layers unfrozen (AST)

    # ── Mel spectrogram ─────────────────────────────────────────────
    sample_rate: int = 16000
    n_mels: int = 128
    hop_ms: int = 10
    win_ms: int = 25


@dataclass
class LanguageEncoderConfig:
    backbone: Literal['clip', 'bert', 't5'] = 'clip'
    clip_model: str = 'ViT-B/32'
    t5_model: str = 't5-small'
    bert_model: str = 'bert-base-uncased'
    frozen: bool = True                             # freeze backbone, train only projection
    d_z: int = 256                                  # language context dimension (= d)


@dataclass
class ModelConfig:
    # ── Common embedding dimension (C1) ─────────────────────────────
    d: int = 256                                    # all f_rep^(m) output to ℝ^d

    # ── State updater Ψ ─────────────────────────────────────────────
    psi_backbone: Literal['gru', 'transformer', 'lstm', 'mamba'] = 'gru'
    d_s: int = 512                                  # state dimension (= 2d)
    psi_layers: int = 1                             # GRU/LSTM layers
    psi_transformer_heads: int = 8                  # Transformer heads
    psi_transformer_layers: int = 4                 # Transformer layers

    # ── Shared MLP ──────────────────────────────────────────────────
    d_h: int = 512                                  # shared representation dim (= d_s)
    shared_mlp_layers: int = 3
    shared_mlp_activation: str = 'gelu'

    # ── Policy networks ─────────────────────────────────────────────
    sigma_max: float = 0.3                          # π^L variance upper bound
    sigma_init: float = 0.2                         # initial π^L variance
    sigma_final: float = 0.05                       # final π^L variance after annealing

    # ── Task head q ─────────────────────────────────────────────────
    task_head_fusion: Literal['concat', 'cross_attn'] = 'concat'
    num_classes: int = 1000                         # override per dataset

    # ── Modalities ──────────────────────────────────────────────────
    modalities: List[int] = field(default_factory=lambda: [1, 2])
    # {1}: vision only, {2}: audio only, {1,2}: joint


@dataclass
class TrainingConfig:
    # ── Stopping distribution ───────────────────────────────────────
    T: int = 4                                      # max perception steps
    stop_lambda: float = 0.5                        # exponential stopping λ
    gamma: float = 0.99                             # RL discount factor

    # ── Learning rates (four separate optimisers) ───────────────────
    lr_rep: float = 1e-4                            # Step 1: representation
    lr_lang_a: float = 5e-5                         # Step 2A: language Path A
    lr_lang_b: float = 1e-5                         # Step 2B: language Path B
    lr_rl: float = 3e-4                             # Step 3: policy
    lr_value: float = 5e-4                          # Step 4: value network
    weight_decay: float = 0.01                      # AdamW for representation

    # ── Gradient clipping ───────────────────────────────────────────
    grad_clip_rep: float = 1.0
    grad_clip_lang_b: float = 0.5
    grad_clip_rl: float = 0.5
    grad_clip_value: float = 1.0

    # ── Curriculum learning ─────────────────────────────────────────
    n_total_steps: int = 100_000                    # ImageNet baseline
    warmup_fraction: float = 0.20                   # Phase I: Steps 1+2A only
    anneal_fraction: float = 0.60                   # Phase II ends at this fraction

    # ── Batch and regularisation ────────────────────────────────────
    batch_size: int = 64
    lambda_align: float = 0.1                       # cross-modal alignment weight
    entropy_reg: float = 0.01                       # π^L entropy regularisation
    entropy_final: float = 0.0                      # linearly decay to 0

    # ── REINFORCE variance reduction ────────────────────────────────
    use_advantage: bool = True                      # subtract value baseline
    normalise_advantages: bool = True               # per-batch mean/std normalise

    # ── Mixed Precision ──────────────────────────────────────────────
    use_amp: bool = True                            # FP16 autocast (CUDA only)


@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    language: LanguageEncoderConfig = field(default_factory=LanguageEncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # ── Experiment metadata ─────────────────────────────────────────
    run_name: str = 'mm_adaptivenn_vit_ast_gru'
    seed: int = 42
    device: str = 'cuda'
    num_workers: int = 0                            # 0=safe on Windows; set 4-8 on Linux
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    output_dir: str = 'outputs/'


def get_config(**overrides) -> Config:
    """Return default config, optionally overriding fields."""
    cfg = Config()
    for k, v in overrides.items():
        parts = k.split('.')
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], v)
    return cfg


# ── Named experiment configurations ─────────────────────────────────────────

def get_vit_ast_gru_config() -> Config:
    """Main result: MM-AdaNN-ViT-AST-GRU"""
    return get_config(
        run_name='mm_adaptivenn_vit_ast_gru',
        **{'encoder.vision_backbone': 'vit_s16',
           'encoder.audio_backbone': 'ast',
           'model.psi_backbone': 'gru'}
    )


def get_vit_ast_trans_config() -> Config:
    """Ablation A7: MM-AdaNN-ViT-AST-Trans"""
    return get_config(
        run_name='mm_adaptivenn_vit_ast_trans',
        **{'encoder.vision_backbone': 'vit_s16',
           'encoder.audio_backbone': 'ast',
           'model.psi_backbone': 'transformer'}
    )


def get_resnet_vggish_gru_config() -> Config:
    """Lightweight baseline: MM-AdaNN-ResNet-VGGish-GRU"""
    return get_config(
        run_name='mm_adaptivenn_resnet_vggish_gru',
        **{'encoder.vision_backbone': 'resnet50',
           'encoder.audio_backbone': 'vggish',
           'model.psi_backbone': 'gru'}
    )


def get_vision_only_config() -> Config:
    """ℳ = {1}: vision-only, degenerates to AdaptiveNN-equivalent"""
    cfg = get_vit_ast_gru_config()
    cfg.model.modalities = [1]
    cfg.run_name = 'mm_adaptivenn_vision_only'
    return cfg


def get_audio_only_config() -> Config:
    """ℳ = {2}: audio-only"""
    cfg = get_vit_ast_gru_config()
    cfg.model.modalities = [2]
    cfg.run_name = 'mm_adaptivenn_audio_only'
    return cfg
