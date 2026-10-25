"""Download + verify + extract MVTec AD metal_nut.

    python scripts/download_dataset.py --data-root data/raw
    python scripts/download_dataset.py --allow-mirror            # if the official link is down
    python scripts/download_dataset.py --verify-archive path/to/metal_nut.tar.xz
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.download import CATEGORY, METAL_NUT_URL, ensure_dataset, verify_against_archive  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--archive-dir", default=str(ROOT / "data" / "archives"),
                    help="where metal_nut.tar.xz lives / is saved")
    ap.add_argument("--url", default=METAL_NUT_URL)
    ap.add_argument("--force", action="store_true", help="re-extract even if dataset exists")
    ap.add_argument("--allow-mirror", action="store_true",
                    help="fall back to an unofficial Hugging Face mirror if the official URL fails")
    ap.add_argument("--verify-archive", default=None,
                    help="SHA-256 compare the extracted tree against an official archive and exit")
    a = ap.parse_args()
    if a.verify_archive:
        rep = verify_against_archive(Path(a.data_root) / CATEGORY, Path(a.verify_archive))
        print(json.dumps({k: (v if isinstance(v, int) else v[:20]) for k, v in rep.items()}, indent=2))
        ok = not (rep["missing_locally"] or rep["extra_locally"] or rep["hash_mismatch"])
        print("IDENTICAL to official archive" if ok else "DIFFERS from official archive")
        sys.exit(0 if ok else 1)
    path = ensure_dataset(Path(a.data_root), Path(a.archive_dir), a.url, a.force, a.allow_mirror)
    print(path)


if __name__ == "__main__":
    main()
