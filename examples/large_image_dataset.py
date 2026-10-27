"""Train on a large image dataset from disk (multi-label), the Efficio way.

Images that don't fit in GPU memory are decoded and resized once into an ImageCache, then
read as uint8 arrays; conversion and augmentation run on the GPU. Works for any image
folder plus a CSV of labels:

    filename,label_a,label_b,label_c        (multi-label: one 0/1 column per label)
    img_0001.png,1,0,1

or a CSV with a single integer "label" column (single-label, uses cross-entropy).

Try it end to end on CPU with generated images:
    python examples/large_image_dataset.py --fake --budget 2m
    python examples/large_image_dataset.py --fake          # resumes, cache is reused

Real data (for example on Colab, with the cache kept on Drive):
    python examples/large_image_dataset.py --images /content/data/images \\
        --labels /content/data/labels.csv --cache /content/drive/MyDrive/cache \\
        --size 224 --grayscale --model densenet121 --budget 2h
"""

import argparse
import csv
import os

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

import efficio
from efficio import CachedImageDataset, ImageCache
from efficio import gpu_augment as A

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", help="folder with the image files")
    p.add_argument("--labels", help="CSV: filename + one 0/1 column per label, or 'label'")
    p.add_argument("--cache", default="./image_cache", help="where the decoded cache goes")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--grayscale", action="store_true", help="store 1 channel (e.g. X-rays)")
    p.add_argument("--io-threads", type=int, default=32,
                   help="parallel file reads for the cache build (raise on slow storage)")
    p.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50", "densenet121"])
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None,
                   help="ImageNet weights (default: on, off with --fake)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--budget", default=None, help='e.g. "2h"; stop and save before it ends')
    p.add_argument("--checkpoint-dir", default="./efficio_checkpoints/large_images")
    p.add_argument("--fake", action="store_true", help="generate a small random dataset")
    args = p.parse_args()
    if args.fake:
        args.images = args.images or "./fake_images"
        args.labels = args.labels or os.path.join(args.images, "labels.csv")
        args.size = min(args.size, 64)
        args.grayscale = True
        args.pretrained = bool(args.pretrained)      # no weight download for the demo
    elif not (args.images and args.labels):
        p.error("--images and --labels are required (or use --fake)")
    if args.pretrained is None:
        args.pretrained = True
    return args


def make_fake_dataset(folder, csv_path, n=300, n_labels=3):
    """n random grayscale PNGs of different sizes with multi-hot labels."""
    from PIL import Image
    if os.path.exists(csv_path):
        return
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        h, w = rng.integers(80, 140, 2)
        name = f"img_{i:04d}.png"
        Image.fromarray(rng.integers(0, 256, (h, w), dtype=np.uint8), "L").save(
            os.path.join(folder, name))
        rows.append([name, *rng.integers(0, 2, n_labels).tolist()])
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows([["filename"] + [f"label_{k}" for k in range(n_labels)]] + rows)
    print(f"generated {n} fake images in {folder}")


def read_labels(images, csv_path):
    """Returns (paths, labels, multi_label). Multi-label: an (N, K) float array."""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    paths = [os.path.join(images, r[0]) for r in rows]
    if header[1:] == ["label"]:
        return paths, np.array([int(r[1]) for r in rows]), False
    return paths, np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float32), True


def make_model(name, n_out, pretrained):
    weights = "DEFAULT" if pretrained else None
    model = getattr(torchvision.models, name)(weights=weights)
    if name.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, n_out)
    else:
        model.fc = nn.Linear(model.fc.in_features, n_out)
    return model


def auroc(scores, targets):
    """Mean ROC AUC over labels that have both classes (rank-based, no sklearn needed)."""
    aucs = []
    for k in range(targets.shape[1]):
        pos, neg = scores[targets[:, k] == 1, k], scores[targets[:, k] == 0, k]
        if len(pos) and len(neg):
            aucs.append(((pos[:, None] > neg[None, :]).float().mean()
                         + 0.5 * (pos[:, None] == neg[None, :]).float().mean()).item())
    return sum(aucs) / len(aucs) if aucs else float("nan")


def main():
    args = parse_args()
    if args.fake:
        make_fake_dataset(args.images, args.labels)
    paths, labels, multi_label = read_labels(args.images, args.labels)

    # 1. Decode and resize once (resumable; returns at once if the cache is already there)
    split = int(0.9 * len(paths))
    size = (args.size, args.size)
    ImageCache.build(paths[:split], labels[:split], os.path.join(args.cache, "train"),
                     size=size, grayscale=args.grayscale, io_threads=args.io_threads)
    ImageCache.build(paths[split:], labels[split:], os.path.join(args.cache, "val"),
                     size=size, grayscale=args.grayscale, io_threads=args.io_threads)

    # 2. Plain DataLoaders over uint8 arrays: no decoding, almost no CPU work
    train = DataLoader(CachedImageDataset(os.path.join(args.cache, "train")),
                       batch_size=args.batch_size, shuffle=True, drop_last=True,
                       num_workers=args.workers)
    val = DataLoader(CachedImageDataset(os.path.join(args.cache, "val")),
                     batch_size=args.batch_size, num_workers=args.workers)

    # 3. Conversion and augmentation on the GPU
    to_model = [A.ToFloat()] + ([A.RepeatChannels(3)] if args.grayscale else [])
    normalize = A.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    augment = A.Compose(to_model + [
        A.RandomResizedCrop(args.size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        A.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        A.RandomHorizontalFlip(),
        A.RandomBrightnessContrast(0.1, 0.1),
        normalize])
    for_eval = A.Compose(to_model + [normalize])

    n_out = labels.shape[1] if multi_label else int(labels.max()) + 1
    model = make_model(args.model, n_out, args.pretrained)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()

    run = efficio.Run(budget=args.budget, checkpoint_dir=args.checkpoint_dir)
    if args.fake:
        run.budget.safety_margin_seconds = 15        # tiny demo budgets
    model, optimizer = run.prepare(model, optimizer)   # resumes if a checkpoint exists

    for x, y in run.loop(train, epochs=args.epochs, gpu_transform=efficio.on_inputs(augment)):
        with run.autocast():
            loss = loss_fn(model(x), y)
        run.backward(loss)
        run.clip_grad_norm(1.0)
        run.step()

    # 4. Validation in fp16/bf16 without gradients
    scores, targets = [], []
    with run.evaluate_mode():
        for x, y in val:
            x = for_eval(x.to(run.device, non_blocking=True))
            out = model(x).float()
            scores.append(torch.sigmoid(out) if multi_label else out.softmax(1))
            targets.append(y.to(run.device))
    scores, targets = torch.cat(scores), torch.cat(targets)
    if multi_label:
        print(f"validation mean AUROC: {auroc(scores, targets):.3f}")
    else:
        print(f"validation accuracy: {(scores.argmax(1) == targets).float().mean().item():.3f}")
    run.report()


if __name__ == "__main__":          # needed for DataLoader workers on Windows/macOS
    main()
