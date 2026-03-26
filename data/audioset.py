"""
data/audioset.py — AudioSet / ESC-50 data loader (ℳ={2})
"""
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class MelSpectrogramTransform:
    def __init__(self, sample_rate=16000, n_mels=128, hop_ms=10, win_ms=25):
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=int(sample_rate * hop_ms / 1000),
            win_length=int(sample_rate * win_ms / 1000),
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform):
        mel = self.transform(waveform)    # (C, F, T)
        return self.db(mel)


class ESC50Dataset(Dataset):
    """
    ESC-50: 50 classes of environmental sounds.
    Expects directory structure: data_root/{class_name}/{file.wav}
    """
    def __init__(self, data_root, split='train', cfg=None, fold=1):
        import csv, glob
        self.files, self.labels = [], []
        self.mel = MelSpectrogramTransform(
            sample_rate=getattr(cfg.encoder, 'sample_rate', 16000) if cfg else 16000,
            n_mels=getattr(cfg.encoder, 'n_mels', 128) if cfg else 128,
        )
        meta_file = os.path.join(data_root, 'meta', 'esc50.csv')
        with open(meta_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                is_val = (int(row['fold']) == fold)
                if (split == 'val') == is_val:
                    self.files.append(os.path.join(data_root, 'audio', row['filename']))
                    self.labels.append(int(row['target']))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        spec = self.mel(wav)                  # (1, 128, T)
        return spec, self.labels[idx]




class AudioSetDataset(torch.utils.data.Dataset):
    """
    AudioSet: large-scale audio classification (527 classes, ~2M clips).
    AudioSet is distributed as pre-extracted features from YouTube.
    Raw audio must be downloaded separately via youtube-dl / yt-dlp.

    Expected structure after downloading:
      data_root/
        balanced_train_segments.csv
        eval_segments.csv
        audio/{youtube_id}.wav

    NOTE: AudioSet has ~50% label noise. ESC-50 is recommended for
    clean evaluation; AudioSet for large-scale pre-training.
    """
    def __init__(self, data_root, split='train', cfg=None, max_samples=None):
        import csv
        csv_file = 'balanced_train_segments.csv' if split == 'train' else 'eval_segments.csv'
        self.files, self.labels = [], []
        self.mel = MelSpectrogramTransform(
            sample_rate=getattr(cfg.encoder, 'sample_rate', 16000) if cfg else 16000,
            n_mels=getattr(cfg.encoder, 'n_mels', 128) if cfg else 128,
        )
        meta_path = os.path.join(data_root, csv_file)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"AudioSet metadata not found: {meta_path}\n"
                "Download from: http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
            )
        with open(meta_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                ytid = row[0].strip()
                # AudioSet labels are comma-separated class indices
                label_str = row[3].strip().strip('"')
                # Use first label for single-label classification
                label = int(label_str.split(',')[0].strip())
                wav_path = os.path.join(data_root, 'audio', f'{ytid}.wav')
                if os.path.exists(wav_path):
                    self.files.append(wav_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        # Pad or trim to 10 seconds
        target_len = 16000 * 10
        if wav.shape[-1] < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[-1]))
        else:
            wav = wav[:, :target_len]
        spec = self.mel(wav)
        return spec, self.labels[idx]


def build_audio_loaders(data_root, cfg, dataset_name='esc50'):
    if dataset_name == 'esc50':
        train_ds = ESC50Dataset(data_root, 'train', cfg)
        val_ds   = ESC50Dataset(data_root, 'val',   cfg)
    elif dataset_name == 'audioset':
        train_ds = AudioSetDataset(data_root, 'train', cfg)
        val_ds   = AudioSetDataset(data_root, 'eval',  cfg)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    def collate(batch):
        specs, labels = zip(*batch)
        X = {2: torch.stack(specs)}
        y = torch.tensor(labels, dtype=torch.long)
        w = ["classify the sound"] * len(labels)
        return X, y, w

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                              shuffle=False, num_workers=cfg.num_workers,
                              collate_fn=collate)
    return train_loader, val_loader
