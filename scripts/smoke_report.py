"""Summarize a GPU smoke test: did each model load and score, where it was placed, how much GPU memory
it used, and roughly how long it will take in the full pilot.

Usage (in the Kaggle notebook, after the smoke test):
    python scripts/smoke_report.py --results results/smoke --models-file configs/pilot_models.txt

Reads the <model>.meta.json, <model>.csv and <model>.error.txt files that score.py writes, prints a
Markdown report and saves it as <results>/smoke_report.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from monbio.scoring import model_slug  # noqa: E402
from score import parse_models_file  # noqa: E402

PILOT_ITEMS = 50
PILOT_ROTATIONS = 4
MEMORY_WARNING = 0.85  # share of a GPU's memory above which the pilot's longer prompts may not fit


def last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "(empty error file)"


def mean_letter_mass(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    with open(csv_path, encoding="utf-8", newline="") as f:
        values = [float(r["letter_mass"]) for r in csv.DictReader(f) if r["nonfinite"] != "True"]
    return sum(values) / len(values) if values else None


def pilot_seconds(meta: dict, pilot_items: int, pilot_rotations: int) -> float | None:
    """Download and loading time, plus the smoke test's scoring time per batch times the pilot's batches.

    Rough: the smoke test has one short batch, the pilot has many, some with longer (Mongolian) prompts.
    """
    if meta.get("load_seconds") is None or meta.get("score_seconds") is None:
        return None  # results from a score.py version that did not record these
    per_item_and_order = meta["n_prompts"] / (meta["n_items"] * meta["rotations"])
    pilot_prompts = per_item_and_order * pilot_items * pilot_rotations
    batch = meta["batch_size"]
    scale = math.ceil(pilot_prompts / batch) / math.ceil(meta["n_prompts"] / batch)
    return meta["load_seconds"] + meta["score_seconds"] * scale


def find_models(results: Path) -> list[str]:
    found = {json.loads(p.read_text(encoding="utf-8"))["model_id"] for p in results.glob("*.meta.json")}
    found |= {p.name.removesuffix(".error.txt").replace("__", "/") for p in results.glob("*.error.txt")}
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="The --out folder of the smoke test")
    parser.add_argument("--models-file", help="Report these models, also those that did not run (default: models in --results)")
    parser.add_argument("--pilot-items", type=int, default=PILOT_ITEMS)
    parser.add_argument("--pilot-rotations", type=int, default=PILOT_ROTATIONS)
    args = parser.parse_args()

    results = Path(args.results)
    if not results.is_dir():
        sys.exit(f"{results} not found")
    model_ids = [m for m, _ in parse_models_file(args.models_file)] if args.models_file else find_models(results)

    rows, notes = [], []
    total, n_estimated = 0.0, 0
    gpus, gpu_sizes = None, []
    for model_id in model_ids:
        slug = model_slug(model_id)
        meta_path, error_path = results / f"{slug}.meta.json", results / f"{slug}.error.txt"
        if not meta_path.exists():
            if error_path.exists():
                result = "FAILED: " + last_line(error_path.read_text(encoding="utf-8"))[:150]
                notes.append(f"{model_id}: failed, the full error is in {error_path}.")
            else:
                result = "not run"
            rows.append([model_id, result] + [""] * 7)
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        env = meta.get("environment") or {}
        sizes = env.get("gpu_memory_gb") or []
        gpus, gpu_sizes = gpus or env.get("device"), gpu_sizes or sizes
        peaks = meta.get("gpu_peak_gb")
        placement = meta.get("device_placement") or {}
        n, bad, no_letter = meta["n_prompts"], meta["n_nonfinite"], meta["n_no_letter"]
        seconds = pilot_seconds(meta, args.pilot_items, args.pilot_rotations)
        if seconds is not None:
            total, n_estimated = total + seconds, n_estimated + 1
        mass = mean_letter_mass(results / f"{slug}.csv")

        off_gpu = [d for d in placement if not d.startswith("cuda")]
        if off_gpu and len(off_gpu) < len(placement):
            notes.append(f"{model_id}: some weights are on {' and '.join(off_gpu)}, not on a GPU, so it will be very slow.")
        if peaks and sizes and any(p > MEMORY_WARNING * s for p, s in zip(peaks, sizes)):
            notes.append(
                f"{model_id}: GPU memory peak is above {MEMORY_WARNING:.0%} of a GPU. The pilot's longer prompts "
                f"may not fit; consider batch_size=4 for it in configs/pilot_models.txt."
            )
        if bad:
            notes.append(
                f"{model_id}: {bad} of {n} prompts gave non-finite scores (float16 overflow?). "
                f"Rerun it with dtype=float32 (see configs/pilot_models.txt)."
            )
        if no_letter:
            notes.append(f"{model_id}: no answer letter within the token budget in {no_letter} of {n} prompts.")
        rows.append([
            model_id,
            "ok",
            "+".join(d.removeprefix("cuda:") for d in placement) or "?",
            " + ".join(f"{p:.1f}" for p in peaks) if peaks else "-",
            f"{meta['load_seconds']:.0f}" if meta.get("load_seconds") is not None else f"{meta['seconds']:.0f} (total)",
            f"{seconds / 60:.0f}" if seconds is not None else "?",
            f"{bad}/{n}",
            f"{no_letter}/{n}",
            f"{mass:.2f}" if mass is not None else "-",
        ])

    header = ["model", "result", "GPUs used", "GPU peak, GB", "load, s", "pilot, min", "non-finite", "no letter", "letter mass"]
    lines = [
        "# GPU smoke test",
        "",
        f"GPUs: {gpus or 'unknown'}" + (f" ({', '.join(f'{s:.1f}' for s in gpu_sizes)} GB)" if gpu_sizes else ""),
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *["| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in rows],
        "",
        "GPUs used: the devices that hold the model's weights (0+1 = split across both GPUs).",
        "load, s: download plus loading. pilot, min: load time plus the smoke test's scoring time per batch,",
        f"scaled to the pilot ({args.pilot_items} items x {args.pilot_rotations} option orders). Rough: the "
        "pilot's batches include longer prompts, which also need more GPU memory than the smoke test did.",
        "",
    ]
    if n_estimated:
        models = "model" if n_estimated == 1 else "models"
        lines += [f"Estimated pilot time for the {n_estimated} {models} that ran: about {total / 3600:.1f} hours.", ""]
    if notes:
        lines += ["## Check these", "", *[f"- {note}" for note in notes], ""]
    report = "\n".join(lines)
    (results / "smoke_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
