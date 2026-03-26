# MultiModal-AdaptiveNN (MM-AdaptiveNN)

Language-guided sequential adaptive perception across visual and auditory modalities.

> Implements the framework described in:
> *"Language-Guided Multimodal Adaptive Neural Networks for Efficient Visual-Auditory Perception"*

---

## Architecture

```
w (language instruction)
    │
    ▼
f_lang ──────────────────────────── z ──────────────────────┐
                                    │                        │
                              FiLM inject                    │
                                    │                        │
                              Shared MLP ◄── sg(s_{t-1})    │
                                    │                        │
                    ┌───────────────┴───────────────┐        │
                    ▼                               ▼        │
                  π^M                             π^L        │
                    │                               │        │
                  m_t                             l_t        │
                    │                               │        │
                    └───────────┬───────────────────┘        │
                                ▼                            │
                    f_rep^(m_t)(X^(m_t), l_t)               │
                                │                            │
                               o_t                          │
                                │                            │
                          Ψ(s_{t-1}, o_t)    [NO z here]    │
                                │                            │
                              s_t ─────────────────────────► q ──► ŷ_t
```

**Key constraints (C1–C6 / AC-1~5):**
- `z` enters only `Shared MLP`, `π^M`, `π^L`, and `q` — NEVER `Ψ` or `f_rep^(m)` (AC-1 / C6)
- `s_{t-1}` is detached in policy and RL computations (AC-2 / C4)
- `z` is pre-computed once per sample before the perception loop (AC-5)
- `θ_π^M ∩ θ_π^L = ∅` (AC-4 / C5)
- All `f_rep^(m)` output to `ℝ^d` via projection heads (C1)

---

## Project Structure

```
mm_adaptivenn/
├── models/
│   ├── language_encoder.py     # f_lang (CLIP / T5), FiLM layers
│   ├── modality_encoders.py    # f_rep^(m): ViT-S/16, ResNet-50, AST, VGGish
│   ├── state_updater.py        # Ψ: GRU, LSTM, Causal Transformer
│   ├── policy_networks.py      # Shared MLP, π^M (categorical), π^L (Gaussian),
│   │                           #   TaskHead q, ValueNetwork V^π
│   └── mm_adaptivenn.py        # Full model, dual-graph perception loop
├── training/
│   ├── trainer.py              # Four-step training algorithm (Theorem 1)
│   │                           #   + TensorBoard logging, Top-5 / E[t_o] metrics
│   ├── curriculum.py           # Three-phase curriculum scheduler
│   └── losses.py               # L_rep, L_lang (Path A+B), L_rl, L_align
├── data/
│   ├── imagenet.py             # ℳ={1}: ImageNet / CUB-200 / FGVC-Aircraft
│   ├── audioset.py             # ℳ={2}: AudioSet / ESC-50
│   ├── audiovisual.py          # ℳ={1,2}: DAVE / CALVIN / AVSBench
│   └── salicon.py              # Visual saliency (SALICON), NSS/AUC metrics
├── configs/
│   └── default.py              # All hyperparameters + named experiment presets
├── utils/
│   └── grad_checks.py          # AC-1~5 runtime verification
├── tests/
│   ├── test_ac_constraints.py  # AC-1~5 structural / parameter-set tests (10 tests)
│   └── test_gradient_flow.py   # Theorem 1 gradient decomposition tests (13 tests)
├── scripts/
│   ├── train.py                # Main training entry point
│   └── evaluate.py             # Evaluation: accuracy / efficiency / attention
├── test_forward.py             # End-to-end smoke test (no dataset required)
├── requirements.txt            # Installation instructions
└── README.md
```

> **Note:** `TaskHead` and `ValueNetwork` are defined in `models/policy_networks.py`
> (not in a separate `task_head.py`). Mel-spectrogram preprocessing is implemented
> inline in `data/audioset.py` and `data/audiovisual.py`.

---

## Installation

```bash
# 1. Core deep learning (match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Encoders
pip install timm                        # ViT-S/16
pip install openai-clip                 # CLIP language encoder
pip install transformers                # AST, T5, HuggingFace CLIP fallback

# 3. Utilities
pip install scipy==1.11.4 pillow numpy  # scipy ≤1.11 avoids OpenBLAS DLL conflict on Windows

# 4. Optional: TensorBoard logging
pip install tensorboard
```

See `requirements.txt` for full version-pinned instructions and Windows-specific notes.

---

## Quick Start

```python
from configs.default import get_config
from models.mm_adaptivenn import MMAdaptiveNN
from training.trainer import Trainer

cfg = get_config()
model = MMAdaptiveNN(cfg)
trainer = Trainer(model, cfg)          # TensorBoard logs → cfg.output_dir/tensorboard/
trainer.train(train_loader, val_loader)
```

---

## Running Tests

```bash
# AC constraint tests (no dataset, no network required)
python -m pytest tests/test_ac_constraints.py -v

# Gradient flow / Theorem 1 verification (no dataset, no network required)
python -m pytest tests/test_gradient_flow.py -v

# All 23 tests
python -m pytest tests/ -v

# End-to-end smoke test (random weights, no download)
python test_forward.py
```

---

## Training

```bash
# Vision-only — ImageNet
python scripts/train.py --modalities vision --dataset imagenet \
    --data_root /data/imagenet --output_dir outputs/run_imagenet

# Audio-only — ESC-50
python scripts/train.py --modalities audio --dataset esc50 \
    --data_root /data/esc50 --output_dir outputs/run_esc50

# Joint audiovisual — DAVE
python scripts/train.py --modalities audiovisual --dataset dave \
    --data_root /data/dave --output_dir outputs/run_dave

# Resume from checkpoint
python scripts/train.py --modalities vision --dataset cub200 \
    --data_root /data/cub200 --resume outputs/run_cub/ckpt_050000.pt \
    --output_dir outputs/run_cub

# Monitor training
tensorboard --logdir outputs/run_imagenet/tensorboard
```

---

## Evaluation

```bash
# Standard accuracy (Top-1 / Top-5) + efficiency E[t_o]
python scripts/evaluate.py --checkpoint outputs/run_imagenet/final.pt \
    --dataset imagenet --data_root /data/imagenet

# Efficiency–accuracy trade-off (sweep λ)
python scripts/evaluate.py --checkpoint outputs/run_imagenet/final.pt \
    --dataset imagenet --data_root /data/imagenet \
    --eval_mode efficiency --lambda_sweep 0.2 0.3 0.5 0.7 0.9

# Human attention comparison (SALICON NSS)
python scripts/evaluate.py --checkpoint outputs/run_imagenet/final.pt \
    --dataset salicon --data_root /data/salicon \
    --eval_mode attention

# Verify AC constraints on trained model
python scripts/evaluate.py --checkpoint outputs/run_imagenet/final.pt \
    --dataset imagenet --data_root /data/imagenet \
    --eval_mode ac_check
```

---

## Modality Configurations

| Config | ℳ | Datasets |
|--------|---|---------|
| `--modalities vision` | {1} | ImageNet, CUB-200, FGVC-Aircraft |
| `--modalities audio` | {2} | AudioSet, ESC-50 |
| `--modalities audiovisual` | {1,2} | DAVE, CALVIN, AVSBench |

---

## Theorem 1 Implementation

The four-step training algorithm implements the gradient decomposition:

```
∇_θ L(θ) = ∇_θ L_rep(θ) + ∇_θ L_lang(θ) + ∇_θ L_rl(θ)
```

| Step | Loss | Parameter group | Graph |
|------|------|-----------------|-------|
| 1 | `L_rep` | `{θ_rep, θ_Ψ, θ_q}` | Full BPTT via `s_full` |
| 2A | `L_lang` Path A | `θ_lang` | Full graph, `z → q → ŷ` |
| 2B | `L_lang` Path B | `θ_lang` | REINFORCE via `sg(s)` |
| 3 | `L_rl` | `{θ_π^M, θ_π^L, θ_shared}` | REINFORCE + reparameterisation |
| 4 | `L_value` | `θ_V` | MSE via `sg(s)` |

**Correctness proofs (automated):** `tests/test_gradient_flow.py` contains 13 pytest
tests (GF-1 through GF-12 + integration) that verify every gradient reach/block claim
above at runtime using real PyTorch autograd.

---

## Named Experiment Configurations

Defined in `configs/default.py`:

| Function | Config | Description |
|----------|--------|-------------|
| `get_vit_ast_gru_config()` | ViT-S/AST/GRU | Main result |
| `get_vit_ast_trans_config()` | ViT-S/AST/Transformer | Ablation A7 |
| `get_resnet_vggish_gru_config()` | ResNet-50/VGGish/GRU | Lightweight baseline |
| `get_vision_only_config()` | ViT-S/–/GRU | ℳ={1} ablation |
| `get_audio_only_config()` | –/AST/GRU | ℳ={2} ablation |
