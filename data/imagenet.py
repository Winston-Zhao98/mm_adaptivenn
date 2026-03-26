"""
data/imagenet.py  — ImageNet / CUB-200 / FGVC-Aircraft data loader (ℳ={1})

Datasets:
  imagenet      — ImageNet-1K (ILSVRC 2012), 1000 classes
                  Structure: data_root/{train,val}/{class_id}/...jpg
  cub200        — CUB-200-2011, 200 bird species
                  Structure: data_root/{train,test}/{class_name}/...jpg
  fgvc_aircraft — FGVC-Aircraft, 100 aircraft variants
                  Uses torchvision.datasets.FGVCAircraft (official split)
                  Download: torchvision auto-downloads to data_root/
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.datasets as D
from PIL import Image


# ── Shared transforms ─────────────────────────────────────────────────────────

TRAIN_TRANSFORM = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── CUB-200-2011 ──────────────────────────────────────────────────────────────

class CUB200Dataset(Dataset):
    """
    CUB-200-2011: 200 bird species, 11,788 images.

    Expected directory structure (standard CUB download):
      data_root/
        images/                      # all images
        train_test_split.txt         # 1 = train, 0 = test
        images.txt                   # id → filename
        image_class_labels.txt       # id → class (1-indexed)
        classes.txt                  # class id → name
    """
    def __init__(self, data_root: str, split: str = 'train',
                 transform=None):
        self.data_root = data_root
        self.transform = transform or (TRAIN_TRANSFORM if split == 'train'
                                       else VAL_TRANSFORM)
        self.files, self.labels = [], []

        # Load id → filename
        id2file = {}
        with open(os.path.join(data_root, 'images.txt')) as f:
            for line in f:
                img_id, fname = line.strip().split()
                id2file[img_id] = fname

        # Load id → class label (0-indexed)
        id2label = {}
        with open(os.path.join(data_root, 'image_class_labels.txt')) as f:
            for line in f:
                img_id, class_id = line.strip().split()
                id2label[img_id] = int(class_id) - 1  # 0-indexed

        # Load train/test split
        with open(os.path.join(data_root, 'train_test_split.txt')) as f:
            for line in f:
                img_id, is_train = line.strip().split()
                include = (split == 'train') == (is_train == '1')
                if include:
                    self.files.append(
                        os.path.join(data_root, 'images', id2file[img_id])
                    )
                    self.labels.append(id2label[img_id])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]


# ── FGVC-Aircraft ─────────────────────────────────────────────────────────────

class FGVCAircraftDataset(Dataset):
    """
    FGVC-Aircraft: 100 aircraft variant classes, ~10,000 images.

    Uses torchvision's official FGVCAircraft dataset class (torchvision >= 0.14).
    Falls back to manual loading for older torchvision versions.

    Official splits: 'train', 'val', 'trainval', 'test'
    Label granularity: 'variant' (100 classes, finest), 'family', 'manufacturer'

    Download: set download=True to auto-download from torchvision CDN.
    """
    def __init__(self, data_root: str, split: str = 'train',
                 annotation_level: str = 'variant',
                 transform=None, download: bool = False):
        self.transform = transform or (TRAIN_TRANSFORM if split == 'train'
                                       else VAL_TRANSFORM)
        try:
            # torchvision >= 0.14 has native FGVCAircraft support
            self._ds = D.FGVCAircraft(
                root=data_root,
                split=split,
                annotation_level=annotation_level,
                transform=self.transform,
                download=download,
            )
            self._use_native = True
        except AttributeError:
            # Older torchvision: fall back to manual loading
            self._use_native = False
            self._init_manual(data_root, split, annotation_level)

    def _init_manual(self, data_root, split, annotation_level):
        """
        Manual loading for torchvision < 0.14.
        Expected structure (standard FGVC-Aircraft download):
          data_root/fgvc-aircraft-2013b/data/
            images/
            images_variant_{train,val,test,trainval}.txt
        """
        data_dir = os.path.join(data_root, 'fgvc-aircraft-2013b', 'data')
        ann_file = os.path.join(
            data_dir, f'images_{annotation_level}_{split}.txt'
        )
        if not os.path.exists(ann_file):
            raise FileNotFoundError(
                f"FGVC-Aircraft annotation file not found: {ann_file}\n"
                "Download from: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"
            )
        # Build class-to-index mapping
        classes = set()
        entries = []
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    img_id, class_name = parts
                    classes.add(class_name)
                    entries.append((img_id, class_name))
        self._class2idx = {c: i for i, c in enumerate(sorted(classes))}
        self._files  = [os.path.join(data_dir, 'images', f'{e[0]}.jpg')
                        for e in entries]
        self._labels = [self._class2idx[e[1]] for e in entries]

    def __len__(self):
        return len(self._ds) if self._use_native else len(self._files)

    def __getitem__(self, idx):
        if self._use_native:
            img, label = self._ds[idx]
            return img, label
        img = Image.open(self._files[idx]).convert('RGB')
        return self.transform(img), self._labels[idx]

    @property
    def class_to_idx(self):
        if self._use_native:
            return self._ds.class_to_idx
        return self._class2idx


# ── Builder ───────────────────────────────────────────────────────────────────

def build_imagenet_loaders(data_root, cfg, dataset_name='imagenet'):
    """
    Build train/val DataLoaders for vision-only benchmarks (ℳ={1}).

    Instructions per dataset default to generic "classify the image".
    For fine-grained datasets, the instruction includes the category type.
    """
    if dataset_name == 'imagenet':
        train_ds = D.ImageFolder(
            os.path.join(data_root, 'train'), TRAIN_TRANSFORM
        )
        val_ds = D.ImageFolder(
            os.path.join(data_root, 'val'), VAL_TRANSFORM
        )
        instruction = "classify the object in the image"

    elif dataset_name == 'cub200':
        train_ds = CUB200Dataset(data_root, 'train')
        val_ds   = CUB200Dataset(data_root, 'test')
        instruction = "identify the bird species in the image"

    elif dataset_name == 'fgvc_aircraft':
        train_ds = FGVCAircraftDataset(data_root, 'trainval',
                                       annotation_level='variant')
        val_ds   = FGVCAircraftDataset(data_root, 'test',
                                       annotation_level='variant')
        instruction = "identify the aircraft variant in the image"

    else:
        raise ValueError(
            f"Unknown vision dataset: {dataset_name}. "
            "Supported: imagenet, cub200, fgvc_aircraft"
        )

    def collate(batch):
        imgs, labels = zip(*batch)
        X = {1: torch.stack(imgs)}
        y = torch.tensor(labels, dtype=torch.long)
        w = [instruction] * len(labels)
        return X, y, w

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    return train_loader, val_loader
