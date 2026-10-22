"""Shared setup for the image-folder benchmark (large images decoded from disk).

Both image_folder_baseline.py and image_folder_efficio.py import from here, so they use the
same images, labels, split, model, image size, batch size, augmentation settings and seed.
This file must not import efficio.

Data: either --fake (generated grayscale PNGs, default 5,000 at 512x512) or a real dataset
given as --images FOLDER --labels CSV, in the same CSV format as
examples/large_image_dataset.py: a "filename" column plus one 0/1 column per label.
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torchvision
from torch import nn

BENCHMARK = "image-folder"
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 2              # what a typical notebook user sets; Efficio may tune it
VAL_FRACTION = 0.1
N_FAKE_LABELS = 5
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# Augmentation, identical in meaning in both scripts (CPU torchvision vs GPU efficio ops):
# resize to IMAGE_SIZE, random resized crop, small affine, horizontal flip, brightness and
# contrast, then 1 -> 3 channels and ImageNet normalization.
CROP_SCALE, CROP_RATIO = (0.8, 1.0), (0.9, 1.1)
AFFINE_DEGREES, AFFINE_TRANSLATE, AFFINE_SCALE = 7, (0.05, 0.05), (0.95, 1.05)
BRIGHTNESS = CONTRAST = 0.1


def add_args(parser):
    parser.add_argument("--fake", action="store_true",
                        help="generate grayscale PNGs instead of using --images/--labels")
    parser.add_argument("--fake-count", type=int, default=5000)
    parser.add_argument("--fake-size", type=int, default=512, help="side of the fake PNGs")
    parser.add_argument("--images", help="folder with the image files")
    parser.add_argument("--labels", help="CSV: filename + one 0/1 column per label")
    parser.add_argument("--data-dir", default="./data/image_folder",
                        help="where fake images (and the Efficio cache) are written")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="training image size")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--pretrained", action="store_true", help="ImageNet weights")


def _fake_image(args):
    """One smooth random grayscale image with fine noise, roughly like a scan."""
    from PIL import Image
    path, size, seed = args
    rng = np.random.default_rng(seed)
    low = Image.fromarray(rng.integers(0, 256, (16, 16), dtype=np.uint8), "L")
    img = np.asarray(low.resize((size, size), Image.BICUBIC), dtype=np.int16)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    Image.fromarray(img, "L").save(path)


def make_fake_dataset(folder, count, size):
    """Writes count PNGs and labels.csv into folder (skipped if already there)."""
    csv_path = os.path.join(folder, "labels.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            if sum(1 for _ in f) - 1 == count:
                print(f"fake dataset: using the existing {count} images in {folder}")
                return csv_path
    os.makedirs(folder, exist_ok=True)
    names = [f"img_{i:05d}.png" for i in range(count)]
    jobs = [(os.path.join(folder, n), size, i) for i, n in enumerate(names)]
    with ProcessPoolExecutor() as pool:
        list(pool.map(_fake_image, jobs, chunksize=32))
    labels = np.random.default_rng(0).integers(0, 2, (count, N_FAKE_LABELS))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename"] + [f"label_{k}" for k in range(N_FAKE_LABELS)])
        w.writerows([n, *row] for n, row in zip(names, labels.tolist()))
    print(f"fake dataset: wrote {count} PNGs ({size}x{size}) to {folder}")
    return csv_path


def load_dataset(args):
    """Returns (train_paths, train_labels, val_paths, val_labels) with float multi-hot labels."""
    if args.fake:
        folder = os.path.join(args.data_dir, "fake_images")
        csv_path = make_fake_dataset(folder, args.fake_count, args.fake_size)
    elif args.images and args.labels:
        folder, csv_path = args.images, args.labels
    else:
        raise SystemExit("use --fake, or --images FOLDER --labels CSV")
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))[1:]
    if args.limit:
        rows = rows[:args.limit]
    paths = [os.path.join(folder, r[0]) for r in rows]
    labels = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float32)
    split = int(len(paths) * (1 - VAL_FRACTION))       # fixed split: same in both scripts
    return paths[:split], labels[:split], paths[split:], labels[split:]


def make_model(n_labels, pretrained):
    model = torchvision.models.resnet18(weights="DEFAULT" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, n_labels)
    return model


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


def mean_auroc(scores, targets):
    """Mean ROC AUC over labels that have both classes (rank-based)."""
    aucs = []
    for k in range(targets.shape[1]):
        pos, neg = scores[targets[:, k] == 1, k], scores[targets[:, k] == 0, k]
        if len(pos) and len(neg):
            aucs.append(((pos[:, None] > neg[None, :]).float().mean()
                         + 0.5 * (pos[:, None] == neg[None, :]).float().mean()).item())
    return sum(aucs) / len(aucs) if aucs else float("nan")


def label_accuracy(scores, targets):
    """Mean per-label accuracy at threshold 0.5, in %."""
    return 100.0 * ((scores > 0.5).float() == targets).float().mean().item()
