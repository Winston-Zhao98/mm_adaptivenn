"""
data/audiovisual.py — DAVE / CALVIN / AVSBench data loader (ℳ={1,2})

Each sample returns:
  X = {1: image_tensor, 2: audio_spectrogram}
  y = label
  w = natural-language instruction / question string
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchaudio


class DAVEDataset(Dataset):
    """
    DAVE (NeurIPS 2025): forced audiovisual QA.
    Each question requires both visual and auditory evidence.
    Expected structure:
      data_root/
        annotations.json        # list of {video_id, question, answer, options}
        frames/{video_id}.jpg   # representative frame
        audio/{video_id}.wav    # corresponding audio
    """
    IMG_TRANSFORM = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    def __init__(self, data_root, split='train'):
        from PIL import Image
        self.Image = Image
        self.data_root = data_root
        with open(os.path.join(data_root, f'{split}_annotations.json')) as f:
            self.samples = json.load(f)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=128, hop_length=160
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        vid = s['video_id']

        # Vision
        img_path = os.path.join(self.data_root, 'frames', f'{vid}.jpg')
        img = self.Image.open(img_path).convert('RGB')
        img_t = self.IMG_TRANSFORM(img)

        # Audio
        aud_path = os.path.join(self.data_root, 'audio', f'{vid}.wav')
        wav, sr = torchaudio.load(aud_path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        spec = self.db(self.mel(wav))              # (1, 128, T)

        # Question as instruction
        w = s['question']
        y = s['answer']                            # int (index into options)

        return img_t, spec, y, w


def _pad_or_trim_spec(spec: torch.Tensor, T_target: int = 100) -> torch.Tensor:
    """Pad or trim spectrogram to fixed time length."""
    T_cur = spec.shape[-1]
    if T_cur >= T_target:
        return spec[..., :T_target]
    else:
        return torch.nn.functional.pad(spec, (0, T_target - T_cur))


def build_audiovisual_loaders(data_root, cfg, dataset_name='dave'):
    if dataset_name == 'dave':
        train_ds = DAVEDataset(data_root, 'train')
        val_ds   = DAVEDataset(data_root, 'val')
    else:
        raise NotImplementedError(f"Dataset {dataset_name} loader not yet implemented. "
                                  "Implement by subclassing the pattern above.")

    def collate(batch):
        imgs, specs, labels, questions = zip(*batch)
        specs = [_pad_or_trim_spec(s) for s in specs]
        X = {
            1: torch.stack(imgs),
            2: torch.stack(specs),
        }
        y = torch.tensor(labels, dtype=torch.long)
        w = list(questions)
        return X, y, w

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size,
                              shuffle=False, num_workers=cfg.num_workers,
                              collate_fn=collate)
    return train_loader, val_loader


# ── CALVIN dataset ────────────────────────────────────────────────────────────

class CALVINDataset(torch.utils.data.Dataset):
    """
    CALVIN: Composing Actions by Learning from Interactive Demonstrations.
    Embodied manipulation task with language instruction + visual observation.

    Audio extension: simulated operation sounds added via MuJoCo audio rendering.
    Expected structure:
      data_root/
        task_D_D/
          training/
            episode_XXXXXX/
              {lang_ann_s3d, lang_ann_clip} (language annotations)
              rgb_static.npy      (image frames, shape [T,3,200,200])
              audio_sim.npy       (simulated audio, shape [T, 16000])  ← custom
              lang_goal.npy       (language goal embedding)
          validation/ ...
    """
    IMG_TRANSFORM = T.Compose([
        T.ToPILImage(), T.Resize(224), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    def __init__(self, data_root, split='training', with_audio=True):
        import glob, numpy as np
        self.episodes = sorted(glob.glob(
            os.path.join(data_root, 'task_D_D', split, 'episode_*')
        ))
        self.with_audio = with_audio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=128, hop_length=160
        )
        self.db = torchaudio.transforms.AmplitudeToDB()
        self.np = np
        if not self.episodes:
            raise FileNotFoundError(
                f"No CALVIN episodes found in {data_root}/task_D_D/{split}/\n"
                "Download from: http://calvin.cs.uni-freiburg.de"
            )

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        np = self.np

        # Load first frame as representative visual input
        frames = np.load(os.path.join(ep, 'rgb_static.npy'))  # (T,3,200,200)
        img = torch.from_numpy(frames[0]).float() / 255.0     # (3,200,200)
        img_t = self.IMG_TRANSFORM(img.permute(1,2,0).numpy())  # (3,224,224)

        # Load language goal
        ann_path = os.path.join(ep, 'lang_ann.npy')
        if os.path.exists(ann_path):
            lang_str = str(np.load(ann_path, allow_pickle=True).item())
        else:
            lang_str = "manipulate the object"

        # Load simulated audio (if available)
        audio_path = os.path.join(ep, 'audio_sim.npy')
        if self.with_audio and os.path.exists(audio_path):
            audio = torch.from_numpy(np.load(audio_path)[0]).float().unsqueeze(0)  # (1, 16000)
            spec = self.db(self.mel(audio))                    # (1, 128, T)
        else:
            # Fallback: silent audio
            spec = torch.zeros(1, 128, 100)

        # Task label: which of 34 CALVIN tasks (use episode index mod 34 as placeholder)
        label = idx % 34

        return img_t, spec, label, lang_str


class AVSBenchDataset(torch.utils.data.Dataset):
    """
    AVSBench: Audio-Visual Segmentation benchmark.
    V1 (single sound source) and V2 (multiple sources).

    Expected structure:
      data_root/
        Single-source/    (or Multi-sources/)
          train.txt       (list of video IDs)
          {video_id}/
            frames/{frame_id}.jpg
            audio.wav
            gt/{frame_id}.png   (binary segmentation mask)
    """
    IMG_TRANSFORM = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    MASK_TRANSFORM = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    def __init__(self, data_root, split='train', version='v1'):
        from PIL import Image
        self.Image = Image
        subdir = 'Single-source' if version == 'v1' else 'Multi-sources'
        split_file = os.path.join(data_root, subdir, f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"AVSBench split file not found: {split_file}\n"
                "Download from: https://github.com/OpenNI/AVSBench"
            )
        with open(split_file) as f:
            self.video_ids = [l.strip() for l in f if l.strip()]
        self.data_root = os.path.join(data_root, subdir)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=128, hop_length=160
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        vid_dir = os.path.join(self.data_root, vid)

        # First frame
        frames = sorted(os.listdir(os.path.join(vid_dir, 'frames')))
        img = self.Image.open(
            os.path.join(vid_dir, 'frames', frames[0])
        ).convert('RGB')
        img_t = self.IMG_TRANSFORM(img)

        # Audio
        wav, sr = torchaudio.load(os.path.join(vid_dir, 'audio.wav'))
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
        spec = self.db(self.mel(wav))                          # (1, 128, T)

        # Label: binary (sound source present=1 or not=0)
        label = 1  # AVSBench: positive samples only in dataset

        w = "locate the sounding object in the scene"
        return img_t, spec, label, w


# ── Updated build_audiovisual_loaders ────────────────────────────────────────

def build_audiovisual_loaders(data_root, cfg, dataset_name='dave'):
    if dataset_name == 'dave':
        train_ds = DAVEDataset(data_root, 'train')
        val_ds   = DAVEDataset(data_root, 'val')
    elif dataset_name == 'calvin':
        train_ds = CALVINDataset(data_root, 'training')
        val_ds   = CALVINDataset(data_root, 'validation')
    elif dataset_name == 'avsbench':
        train_ds = AVSBenchDataset(data_root, 'train')
        val_ds   = AVSBenchDataset(data_root, 'val')
    else:
        raise ValueError(f"Unknown audiovisual dataset: {dataset_name}")

    def collate(batch):
        imgs, specs, labels, questions = zip(*batch)
        specs = [_pad_or_trim_spec(s) for s in specs]
        X = {1: torch.stack(imgs), 2: torch.stack(specs)}
        y = torch.tensor(labels, dtype=torch.long)
        w = list(questions)
        return X, y, w

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                              shuffle=False, num_workers=cfg.num_workers,
                              collate_fn=collate)
    return train_loader, val_loader
