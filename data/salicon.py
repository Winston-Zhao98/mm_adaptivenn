"""
data/salicon.py — SALICON visual attention benchmark (ℳ={1})

SALICON is fundamentally different from classification benchmarks:
  - Labels are SALIENCY MAPS (eye-fixation density maps), not class indices
  - Evaluation metrics: KL divergence, NSS, AUC-J, CC, SIM
  - Used for human-behaviour comparison (arch_benchmark.docx §2.7)

Purpose in MM-AdaptiveNN:
  Compare the model's fixation sequence {l_t} against human saliency maps.
  A high NSS / AUC-J means the model attends to visually salient regions,
  matching human perceptual behaviour — validates the cognitive science motivation.

Expected structure (standard SALICON download from LSUN Challenge):
  data_root/
    images/
      train/{img_id}.jpg
      val/{img_id}.jpg
    maps/
      train/{img_id}.png      # ground-truth saliency map (float32 heatmap)
      val/{img_id}.png
    fixations/
      train/{img_id}.mat      # raw fixation points (optional, for NSS)
      val/{img_id}.mat

Download: http://salicon.net/challenge-2017/

Reference:
  Jiang et al. (2015) SALICON: Saliency in Context. CVPR.
"""
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


# ── Transforms ────────────────────────────────────────────────────────────────

IMG_TRANSFORM = T.Compose([
    T.Resize((480, 640)),           # SALICON standard resolution
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_saliency_map(path: str,
                       target_h: int = 480,
                       target_w: int = 640) -> torch.Tensor:
    """
    Load and normalise saliency map to [0, 1] float tensor.
    Returns: (1, H, W)
    """
    smap = np.array(Image.open(path).convert('L'), dtype=np.float32)
    smap = smap / (smap.max() + 1e-8)                  # normalise to [0,1]
    smap_t = torch.from_numpy(smap).unsqueeze(0)       # (1, H, W)
    if smap_t.shape[-2:] != (target_h, target_w):
        smap_t = F.interpolate(
            smap_t.unsqueeze(0), size=(target_h, target_w),
            mode='bilinear', align_corners=False
        ).squeeze(0)
    return smap_t


def _load_fixation_map(path: str,
                       target_h: int = 480,
                       target_w: int = 640) -> torch.Tensor:
    """
    Load raw fixation points from .mat file and convert to binary fixation map.
    Returns: (1, H, W) binary tensor
    """
    try:
        from scipy.io import loadmat
        data = loadmat(path)
        # SALICON .mat contains field 'gaze' with fixation coordinates
        gaze = data.get('gaze', data.get('fixations', None))
        if gaze is None:
            return torch.zeros(1, target_h, target_w)
        fmap = np.zeros((target_h, target_w), dtype=np.float32)
        for row in gaze:
            if hasattr(row, '__iter__'):
                for fix in row:
                    if hasattr(fix, '__iter__') and len(fix) >= 2:
                        x, y = int(fix[0]), int(fix[1])
                        x = max(0, min(x, target_w - 1))
                        y = max(0, min(y, target_h - 1))
                        fmap[y, x] = 1.0
        return torch.from_numpy(fmap).unsqueeze(0)
    except Exception:
        return torch.zeros(1, target_h, target_w)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SALICONDataset(Dataset):
    """
    SALICON dataset for visual saliency prediction and fixation modelling.

    __getitem__ returns:
      img:      (3, H, W) normalised image
      smap:     (1, H, W) ground-truth saliency map, values in [0,1]
      fmap:     (1, H, W) binary fixation map (if .mat files available)
      img_id:   str image identifier
    """
    def __init__(self, data_root: str, split: str = 'train',
                 img_h: int = 480, img_w: int = 640,
                 load_fixations: bool = True):
        self.data_root = data_root
        self.split = split
        self.img_h = img_h
        self.img_w = img_w
        self.load_fixations = load_fixations

        img_dir = os.path.join(data_root, 'images', split)
        map_dir = os.path.join(data_root, 'maps', split)

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"SALICON images not found: {img_dir}\n"
                "Download from: http://salicon.net/challenge-2017/\n"
                "Expected structure: data_root/images/{train,val}/*.jpg\n"
                "                   data_root/maps/{train,val}/*.png"
            )

        img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                           glob.glob(os.path.join(img_dir, '*.JPEG')))
        self.samples = []
        for img_path in img_paths:
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            map_path = os.path.join(map_dir, f'{img_id}.png')
            if os.path.exists(map_path):
                fix_path = os.path.join(
                    data_root, 'fixations', split, f'{img_id}.mat'
                ) if load_fixations else None
                self.samples.append((img_path, map_path, fix_path, img_id))

        if not self.samples:
            raise FileNotFoundError(
                f"No valid SALICON samples found. "
                f"Check images and maps directories under {data_root}"
            )

        self.img_transform = T.Compose([
            T.Resize((img_h, img_w)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, map_path, fix_path, img_id = self.samples[idx]

        img = Image.open(img_path).convert('RGB')
        img_t = self.img_transform(img)
        smap  = _load_saliency_map(map_path, self.img_h, self.img_w)
        fmap  = (_load_fixation_map(fix_path, self.img_h, self.img_w)
                 if fix_path and os.path.exists(fix_path)
                 else torch.zeros(1, self.img_h, self.img_w))

        return img_t, smap, fmap, img_id


# ── Evaluation metrics ────────────────────────────────────────────────────────

def compute_kl_divergence(pred: torch.Tensor,
                           gt: torch.Tensor,
                           eps: float = 1e-7) -> float:
    """
    KL divergence between predicted and ground-truth saliency maps.
    Both inputs: (H, W) or (1, H, W), values in [0,1].
    Lower is better.
    """
    p = pred.float().flatten()
    q = gt.float().flatten()
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return float((q * torch.log(q / (p + eps) + eps)).sum())


def compute_nss(pred_map: torch.Tensor,
                fixation_map: torch.Tensor) -> float:
    """
    Normalised Scanpath Saliency (NSS).
    pred_map:     (H, W) continuous saliency prediction
    fixation_map: (H, W) binary fixation map (1 at fixated pixels)
    Higher is better.
    """
    p = pred_map.float()
    f = fixation_map.float()
    if f.sum() == 0:
        return 0.0
    # Z-score normalise prediction
    p_norm = (p - p.mean()) / (p.std() + 1e-7)
    return float((p_norm * f).sum() / f.sum())


def compute_auc_judd(pred_map: torch.Tensor,
                     fixation_map: torch.Tensor,
                     n_splits: int = 100) -> float:
    """
    AUC-Judd: area under ROC curve using fixation map as positives.
    Higher is better.
    """
    p = pred_map.float().flatten().numpy()
    f = fixation_map.float().flatten().numpy()

    if f.sum() == 0:
        return 0.5

    thresholds = np.linspace(0, np.max(p), n_splits)
    tprs, fprs = [], []
    for t in thresholds:
        tp = ((p >= t) & (f == 1)).sum()
        fp = ((p >= t) & (f == 0)).sum()
        tpr = tp / (f.sum() + 1e-7)
        fpr = fp / ((f == 0).sum() + 1e-7)
        tprs.append(tpr)
        fprs.append(fpr)

    # Trapezoidal integration
    tprs = np.array(tprs[::-1])
    fprs = np.array(fprs[::-1])
    return float(np.trapz(tprs, fprs))


def compute_cc(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Pearson Correlation Coefficient between prediction and ground truth.
    Higher is better (range [-1, 1]).
    """
    p = pred.float().flatten()
    q = gt.float().flatten()
    p = p - p.mean()
    q = q - q.mean()
    denom = (p.std() * q.std() + 1e-7) * p.numel()
    return float((p * q).sum() / denom)


def compute_sim(pred: torch.Tensor, gt: torch.Tensor,
                eps: float = 1e-7) -> float:
    """
    Similarity metric (histogram intersection).
    Higher is better.
    """
    p = pred.float().flatten()
    q = gt.float().flatten()
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return float(torch.minimum(p, q).sum())


class SALICONEvaluator:
    """
    Evaluate saliency predictions against SALICON ground truth.
    Returns dict of all five standard metrics.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._kl  = []
        self._nss = []
        self._auc = []
        self._cc  = []
        self._sim = []

    def update(self, pred_map: torch.Tensor,
               gt_smap: torch.Tensor,
               gt_fmap: torch.Tensor):
        """
        pred_map:  (1, H, W) or (H, W) — model's attention density at final step
        gt_smap:   (1, H, W) — ground-truth saliency
        gt_fmap:   (1, H, W) — ground-truth fixation map
        """
        pred = pred_map.squeeze().cpu()
        smap = gt_smap.squeeze().cpu()
        fmap = gt_fmap.squeeze().cpu()

        self._kl.append(compute_kl_divergence(pred, smap))
        self._nss.append(compute_nss(pred, fmap))
        self._auc.append(compute_auc_judd(pred, fmap))
        self._cc.append(compute_cc(pred, smap))
        self._sim.append(compute_sim(pred, smap))

    def compute(self) -> dict:
        return {
            'KL':  float(np.mean(self._kl)),
            'NSS': float(np.mean(self._nss)),
            'AUC': float(np.mean(self._auc)),
            'CC':  float(np.mean(self._cc)),
            'SIM': float(np.mean(self._sim)),
        }


# ── Fixation predictor from MM-AdaptiveNN trajectory ─────────────────────────

def trajectory_to_saliency_map(
    l_traj: torch.Tensor,          # (T, 2) fixation sequence (x, y) ∈ [0,1]²
    sigma: float = 0.05,
    H: int = 480,
    W: int = 640,
) -> torch.Tensor:
    """
    Convert MM-AdaptiveNN's fixation trajectory into a predicted saliency map
    by placing Gaussians at each fixation location.

    Used to compare model fixations with SALICON human ground truth.
    Returns: (1, H, W) saliency map, normalised to [0,1].
    """
    smap = torch.zeros(H, W)
    ys = torch.arange(H, dtype=torch.float32) / H
    xs = torch.arange(W, dtype=torch.float32) / W
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

    for t in range(l_traj.shape[0]):
        cx, cy = l_traj[t, 0].item(), l_traj[t, 1].item()
        g = torch.exp(-((grid_x - cx)**2 + (grid_y - cy)**2) / (2 * sigma**2))
        smap += g

    # Normalise
    smap = smap / (smap.max() + 1e-8)
    return smap.unsqueeze(0)                               # (1, H, W)


# ── Builder ───────────────────────────────────────────────────────────────────

def build_salicon_loaders(data_root: str, cfg,
                          batch_size: int = 1):
    """
    Build DataLoaders for SALICON.

    Note: batch_size=1 is recommended for evaluation since we need per-image
    trajectory generation from the model's fixation sequence.
    Task instructions are generated per image to prompt spatial attention.
    """
    train_ds = SALICONDataset(data_root, 'train',
                              load_fixations=True)
    val_ds   = SALICONDataset(data_root, 'val',
                              load_fixations=True)

    def collate(batch):
        imgs, smaps, fmaps, img_ids = zip(*batch)
        # X only has vision (ℳ={1})
        X = {1: torch.stack(imgs)}
        # Pack saliency & fixation as auxiliary targets (not used for loss)
        aux = {
            'smap':   torch.stack(smaps),
            'fmap':   torch.stack(fmaps),
            'img_id': list(img_ids),
        }
        # Language instruction — generic spatial attention prompt
        w = ["look at the most important regions in the image"] * len(imgs)
        # y is a dummy label (not used in saliency evaluation)
        y = torch.zeros(len(imgs), dtype=torch.long)
        return X, y, w, aux

    bs = batch_size or getattr(cfg.training, 'batch_size', 1)
    nw = getattr(cfg, 'num_workers', 4)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, collate_fn=collate)
    return train_loader, val_loader
