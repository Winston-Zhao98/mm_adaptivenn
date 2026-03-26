"""
models/modality_encoders.py

Modality encoder family {f_rep^(m)}_{m∈ℳ}.

C1:   All encoders output to ℝ^d via projection heads.
AC-1: Encoders receive ONLY (X^(m), l_t) — never z.
      Verified in utils/grad_checks.py: o_t.grad_fn must not involve θ_lang.

Two-phase perception:
  - Global glimpse (t=0): full input downsampled to glimpse_size
  - Local fixation (t≥1): P×P patch (vision) or Δτ window (audio)

Corresponds to arch_01_encoders.docx.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ── Projection head (implements C1) ──────────────────────────────────────────

class ProjectionHead(nn.Module):
    """Linear(d_raw → d) + LayerNorm. Implements Assumption C1."""
    def __init__(self, d_raw: int, d: int):
        super().__init__()
        self.proj = nn.Linear(d_raw, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


# ── Visual encoder ────────────────────────────────────────────────────────────

class ViTVisualEncoder(nn.Module):
    """
    f_rep^(1) using ViT-S/16.
    Fine-tunes last `finetune_layers` transformer blocks + projection head.
    """
    def __init__(self, d: int, patch_size: int = 96, glimpse_size: int = 64,
                 finetune_layers: int = 4, pretrained: bool = True):
        super().__init__()
        import timm
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=pretrained,
                                      num_classes=0)   # remove classifier
        d_raw = self.vit.embed_dim                      # 384 for ViT-S/16

        # Freeze all layers, then unfreeze last `finetune_layers` blocks
        for p in self.vit.parameters():
            p.requires_grad_(False)
        n_blocks = len(self.vit.blocks)
        for block in self.vit.blocks[n_blocks - finetune_layers:]:
            for p in block.parameters():
                p.requires_grad_(True)
        self.vit.norm.requires_grad_(True)

        self.projection = ProjectionHead(d_raw, d)
        self.patch_size = patch_size
        self.glimpse_size = glimpse_size
        self.d = d

    def extract_patch(self, X: torch.Tensor,
                      l: torch.Tensor) -> torch.Tensor:
        """
        Crop a P×P patch centred at l = (x, y) ∈ [0,1]².
        X: (B, 3, H, W)
        l: (B, 2)  normalised coordinates
        Returns: (B, 3, 224, 224) — resized for ViT
        """
        B, C, H, W = X.shape
        P = self.patch_size
        cx = (l[:, 0] * W).long().clamp(P // 2, W - P // 2)
        cy = (l[:, 1] * H).long().clamp(P // 2, H - P // 2)

        patches = []
        for b in range(B):
            x0, y0 = cx[b] - P // 2, cy[b] - P // 2
            patch = X[b:b+1, :, y0:y0+P, x0:x0+P]
            patch = F.interpolate(patch, size=(224, 224), mode='bilinear',
                                  align_corners=False)
            patches.append(patch)
        return torch.cat(patches, dim=0)

    def global_glimpse(self, X: torch.Tensor) -> torch.Tensor:
        """Downsample full image and encode."""
        x = F.interpolate(X, size=(self.glimpse_size, self.glimpse_size),
                          mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.projection(self.vit(x))

    def forward(self, X: torch.Tensor,
                l: torch.Tensor,
                glimpse: bool = False) -> torch.Tensor:
        """
        AC-1: This function has NO access to z. Input is only (X, l).
        X: (B, 3, H, W)
        l: (B, 2) normalised location, or None for global glimpse
        Returns: o_t ∈ ℝ^{B × d}
        """
        if glimpse:
            return self.global_glimpse(X)
        patch = self.extract_patch(X, l)
        feat = self.vit(patch)                              # (B, 384)
        return self.projection(feat)                        # (B, d)


class ResNetVisualEncoder(nn.Module):
    """
    f_rep^(1) using ResNet-50.
    Ablation A1 backbone variant.
    """
    def __init__(self, d: int, patch_size: int = 96, glimpse_size: int = 64,
                 finetune_layers: int = 1):
        super().__init__()
        import torchvision.models as tvm
        try:
            # torchvision >= 0.13: use weights= API
            weights = tvm.ResNet50_Weights.DEFAULT
            resnet = tvm.resnet50(weights=weights)
        except AttributeError:
            resnet = tvm.resnet50(pretrained=True)  # fallback for older torchvision
        # Freeze up to layer3, fine-tune layer4 + projection
        frozen_modules = [resnet.conv1, resnet.bn1, resnet.layer1,
                          resnet.layer2, resnet.layer3]
        for m in frozen_modules:
            for p in m.parameters():
                p.requires_grad_(False)
        # Keep layer4 trainable
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool,
        )
        self.projection = ProjectionHead(2048, d)
        self.patch_size = patch_size
        self.glimpse_size = glimpse_size
        self.d = d

    def extract_patch(self, X, l):
        B, C, H, W = X.shape
        P = self.patch_size
        cx = (l[:, 0] * W).long().clamp(P // 2, W - P // 2)
        cy = (l[:, 1] * H).long().clamp(P // 2, H - P // 2)
        patches = []
        for b in range(B):
            x0, y0 = cx[b] - P // 2, cy[b] - P // 2
            patch = X[b:b+1, :, y0:y0+P, x0:x0+P]
            patches.append(F.interpolate(patch, (224, 224), mode='bilinear', align_corners=False))
        return torch.cat(patches, dim=0)

    def forward(self, X, l, glimpse=False):
        if glimpse:
            x = F.interpolate(X, (224, 224), mode='bilinear', align_corners=False)
        else:
            x = self.extract_patch(X, l)
        feat = self.backbone(x).flatten(1)
        return self.projection(feat)


# ── Audio encoder ─────────────────────────────────────────────────────────────

class ASTAudioEncoder(nn.Module):
    """
    f_rep^(2) using Audio Spectrogram Transformer (AST).
    Input: mel spectrogram (B, 1, 128, T_frames).
    Window: Δτ seconds centred at l = τ ∈ [0,1].

    AST is structurally symmetric to ViT — both are patch-based Transformers.
    This symmetry is a deliberate design choice (arch_01_encoders.docx §5).
    """
    def __init__(self, d: int, window_sec: float = 1.0,
                 glimpse_size: int = 128, sample_rate: int = 16000,
                 hop_ms: int = 10, finetune_layers: int = 3,
                 pretrained: bool = True):
        super().__init__()
        try:
            from transformers import ASTModel, ASTFeatureExtractor
            if pretrained:
                self.ast = ASTModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
            else:
                from transformers import ASTConfig
                self.ast = ASTModel(ASTConfig())
            d_raw = self.ast.config.hidden_size          # 768
            self._use_transformers = True
        except Exception:
            # Fallback: lightweight dummy for testing
            self.ast = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
            )
            d_raw = 64
            self._use_transformers = False

        # Freeze backbone, unfreeze last N layers
        if self._use_transformers:
            for p in self.ast.parameters():
                p.requires_grad_(False)
            n_enc = len(self.ast.encoder.layer)
            for layer in self.ast.encoder.layer[n_enc - finetune_layers:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

        self.projection = ProjectionHead(d_raw, d)
        self.window_frames = int(window_sec / (hop_ms / 1000))  # frames per window
        self.glimpse_size = glimpse_size
        self.sample_rate = sample_rate
        self.d = d

    def extract_window(self, spec: torch.Tensor,
                       l: torch.Tensor) -> torch.Tensor:
        """
        Crop a time window of length window_frames centred at l = τ ∈ [0,1].
        spec: (B, 1, F, T)  where F=128 mel bins, T=total frames
        l:    (B, 1)        normalised time coordinate
        Returns: (B, 1, F, window_frames) resized to (B, 1, 128, 128)
        """
        B, C, F, T = spec.shape
        W = self.window_frames
        ct = (l[:, 0] * T).long().clamp(W // 2, T - W // 2)

        windows = []
        for b in range(B):
            t0 = ct[b] - W // 2
            win = spec[b:b+1, :, :, t0:t0+W]
            win = F.interpolate(win, size=(128, 128), mode='bilinear', align_corners=False)
            windows.append(win)
        return torch.cat(windows, dim=0)

    def global_glimpse(self, spec: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(spec, (128, 128), mode='bilinear', align_corners=False)
        return self._encode(x)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_transformers:
            # HuggingFace ASTModel expects input_values of shape (B, T, F).
            # x enters as (B, 1, F, T) from the spectrogram pipeline;
            # squeeze channel dim → (B, F, T), then transpose → (B, T, F).
            x_flat = x.squeeze(1).transpose(1, 2)          # (B, T, F)
            out = self.ast(input_values=x_flat)
            feat = out.pooler_output                       # (B, 768)
        else:
            feat = self.ast(x)
        return self.projection(feat)

    def forward(self, spec: torch.Tensor,
                l: torch.Tensor,
                glimpse: bool = False) -> torch.Tensor:
        """
        AC-1: No z here.
        spec: (B, 1, 128, T)
        l:    (B, 1) normalised time coordinate
        """
        if glimpse:
            return self.global_glimpse(spec)
        win = self.extract_window(spec, l)
        return self._encode(win)


class VGGishAudioEncoder(nn.Module):
    """
    f_rep^(2) using VGGish.
    Ablation A1 baseline.
    """
    def __init__(self, d: int, window_sec: float = 1.0,
                 glimpse_size: int = 128, hop_ms: int = 10):
        super().__init__()
        try:
            import torchvggish
            self.vggish = torchvggish.vggish()
            self.vggish.embeddings = nn.Identity()         # remove final embedding
            d_raw = 128
        except ImportError:
            # Lightweight fallback
            self.vggish = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten()
            )
            d_raw = 256

        for p in self.vggish.parameters():
            p.requires_grad_(False)

        self.projection = ProjectionHead(d_raw, d)
        self.window_frames = int(window_sec / (hop_ms / 1000))
        self.glimpse_size = glimpse_size
        self.d = d

    def forward(self, spec: torch.Tensor, l: torch.Tensor,
                glimpse: bool = False) -> torch.Tensor:
        if glimpse:
            x = F.interpolate(spec, (self.glimpse_size, self.glimpse_size),
                              mode='bilinear', align_corners=False)
        else:
            B, C, F, T = spec.shape
            W = self.window_frames
            ct = (l[:, 0] * T).long().clamp(W // 2, T - W // 2)
            windows = []
            for b in range(B):
                t0 = ct[b] - W // 2
                win = F.interpolate(spec[b:b+1, :, :, t0:t0+W],
                                    (96, 96), mode='bilinear', align_corners=False)
                windows.append(win)
            x = torch.cat(windows, dim=0)
        feat = self.vggish(x)
        if isinstance(feat, torch.Tensor) and feat.dim() > 2:
            feat = feat.flatten(1)
        return self.projection(feat)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_modality_encoders(cfg) -> nn.ModuleDict:
    """
    Returns a ModuleDict mapping modality index (str) → encoder.
    AC-1 guarantee: no encoder receives z — enforced by interface.
    """
    encoders = {}

    if 1 in cfg.model.modalities:
        if cfg.encoder.vision_backbone == 'vit_s16':
            encoders['1'] = ViTVisualEncoder(
                d=cfg.model.d,
                patch_size=cfg.encoder.vision_patch_size,
                glimpse_size=cfg.encoder.vision_glimpse_size,
                finetune_layers=cfg.encoder.vision_finetune_layers,
                pretrained=(cfg.encoder.vision_pretrained != 'none'),
            )
        elif cfg.encoder.vision_backbone == 'resnet50':
            encoders['1'] = ResNetVisualEncoder(
                d=cfg.model.d,
                patch_size=cfg.encoder.vision_patch_size,
                glimpse_size=cfg.encoder.vision_glimpse_size,
            )

    if 2 in cfg.model.modalities:
        if cfg.encoder.audio_backbone == 'ast':
            encoders['2'] = ASTAudioEncoder(
                d=cfg.model.d,
                window_sec=cfg.encoder.audio_window_sec,
                glimpse_size=cfg.encoder.audio_glimpse_size,
                sample_rate=cfg.encoder.sample_rate,
                hop_ms=cfg.encoder.hop_ms,
                finetune_layers=cfg.encoder.audio_finetune_layers,
                pretrained=(cfg.encoder.audio_pretrained != 'none'),
            )
        elif cfg.encoder.audio_backbone == 'vggish':
            encoders['2'] = VGGishAudioEncoder(
                d=cfg.model.d,
                window_sec=cfg.encoder.audio_window_sec,
                glimpse_size=cfg.encoder.audio_glimpse_size,
                hop_ms=cfg.encoder.hop_ms,
            )

    return nn.ModuleDict(encoders)
