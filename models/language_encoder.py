"""
models/language_encoder.py

Language encoder f_lang and FiLM conditioning heads.

AC-5: z is computed once per sample BEFORE the perception loop.
AC-1 / C6: z is injected ONLY into Shared MLP, π^M, π^L, q.
            It NEVER enters Ψ or any f_rep^(m).

Corresponds to arch_02_psi_lang.docx §5 and theorem_final.docx §1.
"""
import torch
import torch.nn as nn
from typing import Optional


# ── FiLM conditioning ─────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation:
        FiLM(h, z) = γ(z) ⊙ h + β(z)
    where [γ, β] = MLP_film(z).

    Used to inject z into Shared MLP input.
    Parameters belong to the module that owns this FiLM head (θ_shared or θ_q).
    """
    def __init__(self, d_z: int, d_h: int):
        super().__init__()
        self.film_mlp = nn.Sequential(
            nn.Linear(d_z, d_h * 2),
            nn.GELU(),
            nn.Linear(d_h * 2, d_h * 2),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        h: (..., d_h)
        z: (B, d_z)  — broadcast over time if needed
        """
        # Broadcast z to match h's batch dimensions
        while z.dim() < h.dim():
            z = z.unsqueeze(-2)

        params = self.film_mlp(z)                          # (..., 2*d_h)
        gamma, beta = params.chunk(2, dim=-1)              # each (..., d_h)
        return gamma * h + beta


# ── Language encoder backbones ────────────────────────────────────────────────

class CLIPLanguageEncoder(nn.Module):
    """
    Frozen CLIP text encoder (ViT-B/32) + learned linear projection to d_z.
    θ_lang = {projection weight, projection bias, FiLM heads downstream}.
    Backbone parameters are frozen (not in θ_lang).
    """
    def __init__(self, d_z: int = 256, model_name: str = 'ViT-B/32'):
        super().__init__()
        try:
            import clip
            # clip.load may raise RuntimeError on first run (model download)
            # or any network-related exception; treat all failures as fallback.
            self.clip_model, _ = clip.load(model_name, device='cpu')
            d_raw = self.clip_model.text_projection.shape[1]   # 512 for ViT-B/32
            self._use_transformers = False
        except Exception:
            # Fallback: use HuggingFace transformers CLIPTextModel
            from transformers import CLIPTextModel, CLIPTokenizer
            self.clip_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
            self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
            d_raw = 512
            self._use_transformers = True

        # Freeze backbone
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        # Learned projection (θ_lang parameters)
        self.projection = nn.Sequential(
            nn.Linear(d_raw, d_z),
            nn.LayerNorm(d_z),
        )
        self.d_z = d_z

    def forward(self, w: list) -> torch.Tensor:
        """
        w: list of B instruction strings
        Returns z: (B, d_z)
        """
        if self._use_transformers:
            inputs = self.tokenizer(w, return_tensors='pt', padding=True,
                                    truncation=True, max_length=77)
            inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.clip_model(**inputs)
            raw = out.pooler_output                         # (B, 512)
        else:
            import clip
            tokens = clip.tokenize(w, truncate=True).to(next(self.parameters()).device)
            with torch.no_grad():
                raw = self.clip_model.encode_text(tokens).float()  # (B, 512)

        z = self.projection(raw)                            # (B, d_z)
        return z


class T5LanguageEncoder(nn.Module):
    """
    Frozen T5-small encoder + linear projection.
    Used in ablation A6 to validate language backbone agnosticism.
    """
    def __init__(self, d_z: int = 256, model_name: str = 't5-small'):
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        d_raw = self.encoder.config.d_model                # 512 for t5-small

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.projection = nn.Sequential(
            nn.Linear(d_raw, d_z),
            nn.LayerNorm(d_z),
        )
        self.d_z = d_z

    def forward(self, w: list) -> torch.Tensor:
        inputs = self.tokenizer(w, return_tensors='pt', padding=True,
                                truncation=True, max_length=128)
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.encoder(**inputs)
        # Mean-pool over sequence
        raw = out.last_hidden_state.mean(dim=1)             # (B, d_raw)
        return self.projection(raw)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_language_encoder(cfg) -> nn.Module:
    """
    Build f_lang from config.
    AC-5: caller is responsible for calling this ONCE per sample before perception loop.
    """
    if cfg.language.backbone == 'clip':
        return CLIPLanguageEncoder(d_z=cfg.language.d_z,
                                   model_name=cfg.language.clip_model)
    elif cfg.language.backbone == 't5':
        return T5LanguageEncoder(d_z=cfg.language.d_z,
                                  model_name=cfg.language.t5_model)
    else:
        raise ValueError(f"Unknown language backbone: {cfg.language.backbone}")
