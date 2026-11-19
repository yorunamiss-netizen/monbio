"""Write post-editing sheets (one .xlsx per batch of 50 items). The answer key is left out.

Usage:
    python scripts/make_translation_batches.py            # all batches
    python scripts/make_translation_batches.py --batches 1-3

Existing batch files are never overwritten unless you pass --overwrite, so edited work is safe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monbio.translation import write_batch_sheet  # noqa: E402
from monbio.utils import DATA_DIR, read_jsonl  # noqa: E402


def parse_range(text: str, maximum: int) -> list[int]:
    if text == "all":
        return list(range(1, maximum + 1))
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", default=str(DATA_DIR / "items_en.jsonl"))
    parser.add_argument("--mt", default=str(DATA_DIR / "mt" / "mt_segments.jsonl"))
    parser.add_argument("--out-dir", default=str(DATA_DIR / "translation"))
    parser.add_argument("--batches", default="all", help="e.g. 1-3 or 1,2,5 or all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    items = read_jsonl(args.items)
    if not Path(args.mt).exists():
        sys.exit(f"{args.mt} not found. Run scripts/import_mt.py first.")
    mt = {row["seg_id"]: row["mt_raw"] for row in read_jsonl(args.mt)}
    n_batches = max(it["batch"] for it in items)
    for b in parse_range(args.batches, n_batches):
        members = sorted((it for it in items if it["batch"] == b), key=lambda it: it["order"])
        if not members:
            print(f"batch {b}: no items, skipped")
            continue
        path = Path(args.out_dir) / f"batch_{b:02d}.xlsx"
        if path.exists() and not args.overwrite:
            print(f"batch {b}: {path} exists, left unchanged")
            continue
        n = write_batch_sheet(members, mt, path, batch=b)
        print(f"batch {b}: {len(members)} items, {n} rows -> {path}")


if __name__ == "__main__":
    main()
