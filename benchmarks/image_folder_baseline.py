"""Baseline for the image-folder benchmark: plain PyTorch, fp32, no Efficio.

Every epoch, DataLoader workers open each PNG, decode it, resize it and augment it on the
CPU, the usual torchvision way. Run from the repo root:
    python benchmarks/image_folder_baseline.py --fake --epochs 6
    python benchmarks/image_folder_baseline.py --images DIR --labels labels.csv

Keep this script plain. Do not add mixed precision, cudnn.benchmark, pin_memory, a cache or
any other speed-up here; it is the reference the Efficio run is compared against. Only the
timing (epoch time, time spent waiting for data) is measured here.
"""

import argparse
import time

import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

import image_common as ic
from common import GpuUtilSampler, save_results, sync


class ImageFiles(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            x = self.transform(img.convert("L"))
        return x, torch.from_numpy(self.labels[i])


def main():
    parser = argparse.ArgumentParser()
    ic.add_args(parser)
    args = parser.parse_args()
    s = args.size

    train_paths, train_labels, val_paths, val_labels = ic.load_dataset(args)
    train_tf = T.Compose([
        T.Resize((s, s)),
        T.RandomResizedCrop(s, scale=ic.CROP_SCALE, ratio=ic.CROP_RATIO),
        T.RandomAffine(ic.AFFINE_DEGREES, translate=ic.AFFINE_TRANSLATE, scale=ic.AFFINE_SCALE),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=ic.BRIGHTNESS, contrast=ic.CONTRAST),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(ic.IMAGENET_MEAN, ic.IMAGENET_STD),
    ])
    val_tf = T.Compose([T.Resize((s, s)), T.Grayscale(num_output_channels=3), T.ToTensor(),
                        T.Normalize(ic.IMAGENET_MEAN, ic.IMAGENET_STD)])

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(ImageFiles(train_paths, train_labels, train_tf),
                              batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=ic.NUM_WORKERS)
    val_loader = DataLoader(ImageFiles(val_paths, val_labels, val_tf),
                            batch_size=2 * args.batch_size, num_workers=ic.NUM_WORKERS)
    model = ic.make_model(train_labels.shape[1], args.pretrained).to(device)
    optimizer = ic.make_optimizer(model)
    loss_fn = nn.BCEWithLogitsLoss()

    epoch_times, wait_s, train_s = [], 0.0, 0.0
    with GpuUtilSampler() as sampler:
        for epoch in range(args.epochs):
            model.train()
            sync()
            start = time.perf_counter()
            batches = iter(train_loader)
            while True:
                t0 = time.perf_counter()
                batch = next(batches, None)
                wait_s += time.perf_counter() - t0
                if batch is None:
                    break
                x, y = batch[0].to(device), batch[1].to(device)
                loss = loss_fn(model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            sync()
            epoch_times.append(time.perf_counter() - start)
            train_s += epoch_times[-1]
            print(f"epoch {epoch + 1}/{args.epochs}: {epoch_times[-1]:.1f} s, "
                  f"loss {loss.item():.3f}")

    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            scores.append(torch.sigmoid(model(x.to(device))).cpu())
            targets.append(y)
    scores, targets = torch.cat(scores), torch.cat(targets)
    save_results("imagefolder-baseline", epoch_times,
                 len(train_loader) * args.batch_size, sampler.average,
                 ic.label_accuracy(scores, targets),
                 extra={"benchmark": ic.BENCHMARK, "precision": "fp32",
                        "batch_size": args.batch_size, "image_size": s,
                        "train_images": len(train_paths), "fake_data": args.fake,
                        "data_wait_pct": round(100 * wait_s / train_s, 1),
                        "val_auroc": round(ic.mean_auroc(scores, targets), 4),
                        "cache_build_s": 0.0})


if __name__ == "__main__":          # needed for DataLoader workers on Windows/macOS
    main()
