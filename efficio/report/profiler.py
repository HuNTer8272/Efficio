"""Track where training time goes: setup, waiting for data, and the training steps themselves.
Also samples GPU utilization with nvidia-smi. Phase 2 (Person B).

This module must not import torch, so TensorFlow/Keras support can reuse it.
Run passes it plain numbers (seconds).
"""

import subprocess
import threading
import time

DATA_WAIT_TIP_PCT = 10.0    # above this, the GPU is waiting for data
LOW_UTIL_PCT = 85.0         # below this (with little data-wait), the GPU has spare capacity
SETUP_TIP_PCT = 10.0        # setup above this share of the total is worth pointing out


class Profiler:
    """Times one run. Call first_batch() when the first trained batch arrives, add_step()
    for every later batch, and finish() when the loop ends."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.created_at = clock()
        self.first_batch_at = None
        self.finished_at = None
        self.data_wait_s = 0.0
        self.step_s = 0.0
        self.steps = 0

    def first_batch(self, now=None):
        if self.first_batch_at is None:
            self.first_batch_at = self.clock() if now is None else now

    def add_step(self, data_wait_s, step_s):
        self.data_wait_s += data_wait_s
        self.step_s += step_s
        self.steps += 1

    def finish(self, now=None):
        self.finished_at = self.clock() if now is None else now

    @property
    def setup_s(self):
        end = self.first_batch_at if self.first_batch_at is not None else self._now()
        return end - self.created_at

    @property
    def train_s(self):
        if self.first_batch_at is None:
            return 0.0
        return self._now() - self.first_batch_at

    @property
    def total_s(self):
        return self.setup_s + self.train_s

    @property
    def data_wait_pct(self):
        return 100.0 * self.data_wait_s / self.train_s if self.train_s > 0 else 0.0

    @property
    def setup_pct(self):
        return 100.0 * self.setup_s / self.total_s if self.total_s > 0 else 0.0

    def _now(self):
        return self.finished_at if self.finished_at is not None else self.clock()


class GpuUtilSampler:
    """Samples GPU 0 utilization with nvidia-smi in a daemon thread. Never raises:
    if nvidia-smi fails it skips that sample, and if it is missing it stops trying."""

    CMD = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"]

    def __init__(self, interval=1.0):
        self.interval = interval
        self.samples = []
        self.available = True
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not self.available:
            return
        self._stop.clear()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            # Short wait only: a slow nvidia-smi call must not hold up the end of training
            self._thread.join(timeout=1)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(self.CMD, capture_output=True, text=True, timeout=5)
                self.samples.append(float(out.stdout.strip().splitlines()[0]))
            except FileNotFoundError:   # no nvidia-smi on this machine
                self.available = False
                return
            except Exception:           # anything else: skip this sample, keep training
                pass
            self._stop.wait(self.interval)

    @property
    def average(self):
        return sum(self.samples) / len(self.samples) if self.samples else None


GPU_LOADER_EXAMPLE = (
    "from efficio import GPULoader\n"
    "from efficio.speed import gpu_augment as A\n"
    "loader = GPULoader.from_torchvision(train_set, batch_size=128, transform=A.Compose(\n"
    "    [A.RandomCrop(32, padding=4), A.RandomHorizontalFlip(), A.Normalize(mean, std)]))")


def tips(data_wait_pct, gpu_util_pct, setup_pct, on_gpu=True, gpu_loader_fits=False):
    """Plain-language advice from the measured numbers. gpu_util_pct may be None (no GPU data).
    The data-wait and utilization tips are about keeping a GPU busy, so they need on_gpu.
    gpu_loader_fits: the training set is a torchvision dataset small enough for GPULoader."""
    out = []
    if on_gpu and data_wait_pct > DATA_WAIT_TIP_PCT and gpu_loader_fits:
        out.append(f"The GPU is waiting for data {data_wait_pct:.0f}% of the time. Your dataset "
                   "fits in GPU memory, so keep it there and augment on the GPU:\n"
                   + GPU_LOADER_EXAMPLE)
    elif on_gpu and data_wait_pct > DATA_WAIT_TIP_PCT:
        out.append(f"The GPU is waiting for data {data_wait_pct:.0f}% of the time. Try more "
                   "DataLoader workers, or lighter CPU preprocessing.")
    elif on_gpu and gpu_util_pct is not None and gpu_util_pct < LOW_UTIL_PCT:
        out.append(f"GPU utilization is only {gpu_util_pct:.0f}% and data loading is not the "
                   "bottleneck. A larger batch size may use the GPU better.")
    if setup_pct > SETUP_TIP_PCT:
        out.append(f"Setup took {setup_pct:.0f}% of this run. Downloads or preprocessing are "
                   "using your quota; a data cache is planned in Efficio.")
    return out
