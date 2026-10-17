"""Print all benchmark results and the Efficio speedup per platform.

Run from the repo root:
    python benchmarks/compare.py
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

COLUMNS = [
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
]


def load_results():
    results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            results.append(json.loads(path.read_text()))
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
    print()
    for platform in sorted({r["platform"] for r in results}):
        latest = {}
        for name in ("baseline", "efficio"):
            runs = [r for r in results if r["platform"] == platform and r["run"] == name]
            if runs:
                latest[name] = max(runs, key=lambda r: r.get("timestamp", ""))
        if len(latest) < 2:
            missing = "baseline" if "baseline" not in latest else "efficio"
            print(f"{platform}: no {missing} run yet, can't compute a speedup")
            continue
        base, eff = latest["baseline"]["epochs_per_hour"], latest["efficio"]["epochs_per_hour"]
        print(f"{platform}: efficio {eff} vs baseline {base} epochs/hour "
              f"-> {eff / base:.2f}x speedup")


if __name__ == "__main__":
    results = load_results()
    if not results:
        print(f"No results in {RESULTS_DIR}. Run the benchmark scripts first.")
    else:
        print_table(results)
        print_speedups(results)
