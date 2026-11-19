"""Write every English segment to one CSV for machine translation in Google Sheets.

Usage:
    python scripts/make_mt_sheet.py

Then follow docs/TRANSLATION_GUIDE.md, step 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monbio.translation import write_mt_input  # noqa: E402
from monbio.utils import DATA_DIR, read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", default=str(DATA_DIR / "items_en.jsonl"))
    parser.add_argument("--out", default=str(DATA_DIR / "mt" / "mt_input.csv"))
    args = parser.parse_args()

    items = read_jsonl(args.items)
    n = write_mt_input(items, args.out)
    print(f"Wrote {n} segments from {len(items)} items to {args.out}")
    print('Next: import it into Google Sheets, put =GOOGLETRANSLATE(B2,"en","mn") in C2, fill down,')
    print("paste values only, download as CSV, then run scripts/import_mt.py --csv <file>.")


if __name__ == "__main__":
    main()
