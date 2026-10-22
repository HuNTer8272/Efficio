"""Print all benchmark results and the Efficio speedup per benchmark and platform.

Run from the repo root:
    python benchmarks/compare.py
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_BENCHMARK = "resnet18-cifar10"      # results saved before the "benchmark" field

COLUMNS = [
    ("benchmark", "benchmark"),
    ("run", "run"),
    ("platform", "platform"),
    ("gpu", "gpu"),
    ("precision", "precision"),
    ("avg epoch s", "avg_epoch_s_excl_first"),
    ("epochs/hour", "epochs_per_hour"),
    ("images/s", "images_per_s"),
    ("peak mem GB", "peak_mem_gb"),
    ("GPU util %", "avg_gpu_util_pct"),
    ("test acc %", "test_acc_pct"),
    ("data-wait %", "data_wait_pct"),
    ("cache build s", "cache_build_s"),
]


def load_results():
    results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            r = json.loads(path.read_text())
            r.setdefault("benchmark", DEFAULT_BENCHMARK)
            results.append(r)
        except (OSError, json.JSONDecodeError) as e:
            print(f"skipping {path.name}: {e}")
    return results


def print_table(results):
    rows = [[str(r.get(key, "")) for _, key in COLUMNS] for r in results]
    headers = [name for name, _ in COLUMNS]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def print_speedups(results):
    """Speedup of the latest run of each kind over the latest run of each baseline (a run
    name containing "baseline", e.g. imagefolder-baseline-4workers), per benchmark and
    platform."""
    print()
    for bench, platform in sorted({(r["benchmark"], r["platform"]) for r in results}):
        latest = {}
        for r in results:
            if (r["benchmark"], r["platform"]) == (bench, platform):
                if r["run"] not in latest or r.get("timestamp", "") > latest[r["run"]].get("timestamp", ""):
                    latest[r["run"]] = r
        bases = sorted(name for name in latest if "baseline" in name)
        others = sorted(name for name in latest if "baseline" not in name)
        if not bases or not others:
            missing = "baseline" if not bases else "efficio"
            print(f"{bench} / {platform}: no {missing} run yet, can't compute a speedup")
            continue
        for base_name in bases:
            base = latest[base_name]["epochs_per_hour"]
            label = "baseline" if len(bases) == 1 else base_name
            for name in others:
                eff = latest[name]["epochs_per_hour"]
                print(f"{bench} / {platform}: {name} {eff} vs {label} {base} epochs/hour "
                      f"-> {eff / base:.2f}x speedup")


if __name__ == "__main__":
    results = load_results()
    if not results:
        print(f"No results in {RESULTS_DIR}. Run the benchmark scripts first.")
    else:
        results.sort(key=lambda r: (r["benchmark"], r["platform"], r["run"]))
        print_table(results)
        print_speedups(results)
