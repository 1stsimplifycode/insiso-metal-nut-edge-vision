"""Download, validate and safely extract the MVTec AD `metal_nut` category.

Standard library only, so it runs on Colab, Windows and Linux before any
project dependency is installed.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

METAL_NUT_URL = (
    "https://www.mydrive.ch/shares/150459/c68856a21dca589b0f8ff6d4ee0f18f4/"
    "download/420937637-1629959294/metal_nut.tar.xz"
)
ARCHIVE_NAME = "metal_nut.tar.xz"
CATEGORY = "metal_nut"
EXPECTED_SUBDIRS = ("train", "test", "ground_truth")
CHUNK = 1 << 20


def log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def dataset_is_complete(root: Path) -> bool:
    """True when <root>/metal_nut has the three expected sub-directories with images."""
    cat = root / CATEGORY
    if not all((cat / d).is_dir() for d in EXPECTED_SUBDIRS):
        return False
    return any((cat / "train" / "good").glob("*.png")) and any((cat / "test").rglob("*.png"))


def _remote_size(url: str, timeout: float) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "insiso-downloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def download(url: str, dest: Path, retries: int = 5, timeout: float = 60.0) -> Path:
    """Resumable download to `dest` via a `.part` file. Raises on HTTP errors / short reads."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    total = _remote_size(url, timeout)
    if total:
        log(f"remote size: {total / 1e6:.1f} MB")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if total and have == total:
            break
        headers = {"User-Agent": "insiso-downloader/1.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
            log(f"resuming at {have / 1e6:.1f} MB (attempt {attempt})")
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if have and r.status != 206:
                    log("server ignored Range request; restarting from zero")
                    have = 0
                mode = "ab" if have else "wb"
                expected = total or (have + int(r.headers.get("Content-Length", 0)) or None)
                t0, last = time.time(), 0.0
                with open(part, mode) as f:
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        f.write(buf)
                        have += len(buf)
                        if time.time() - last > 2:
                            rate = (have / 1e6) / max(time.time() - t0, 1e-6)
                            pct = f"{100 * have / expected:5.1f}%" if expected else "?"
                            log(f"{pct}  {have / 1e6:7.1f} MB  ({rate:.1f} MB/s)")
                            last = time.time()
            if total and have != total:
                raise IOError(f"short read: {have} of {total} bytes")
            break
        except urllib.error.HTTPError as e:
            if e.code == 416 and total and have == total:
                break
            log(f"HTTP error {e.code}: {e.reason}")
            if e.code in (401, 403, 404):
                raise
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            log(f"network error: {e}")
        if attempt == retries:
            raise RuntimeError(f"download failed after {retries} attempts; partial file kept at {part}")
        time.sleep(min(2 ** attempt, 30))

    if total and part.stat().st_size != total:
        raise RuntimeError("partial download: size mismatch")
    os.replace(part, dest)
    log(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def validate_archive(archive: Path) -> list[tarfile.TarInfo]:
    """Open the full xz stream (detects truncation/corruption) and vet every member path."""
    log("validating archive (full decompression pass)...")
    with tarfile.open(archive, "r:xz") as tf:
        members = tf.getmembers()  # reads through the whole stream
    bad = []
    for m in members:
        p = Path(m.name)
        if p.is_absolute() or ".." in p.parts or m.name.startswith(("/", "\\")):
            bad.append(m.name)
        elif m.issym() or m.islnk() or m.isdev():
            bad.append(m.name)
        elif p.parts and p.parts[0] != CATEGORY:
            bad.append(m.name)
    if bad:
        raise RuntimeError(f"unsafe archive members refused: {bad[:5]}")
    n_png = sum(1 for m in members if m.isfile() and m.name.endswith(".png"))
    log(f"archive OK: {len(members)} members, {n_png} PNG files, all paths inside '{CATEGORY}/'")
    return members


def safe_extract(archive: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    root_resolved = out_root.resolve()
    with tarfile.open(archive, "r:xz") as tf:
        for m in tf.getmembers():
            target = (out_root / m.name).resolve()
            if root_resolved not in target.parents and target != root_resolved:
                raise RuntimeError(f"path traversal blocked: {m.name}")
        if sys.version_info >= (3, 12):
            tf.extractall(out_root, filter="data")
        else:
            tf.extractall(out_root)
    log(f"extracted to {out_root / CATEGORY}")


# Unofficial fallback: per-file mirror on the Hugging Face Hub. Used only when the official
# archive is unreachable; provenance records it, and `verify_against_archive` lets a later
# official download confirm byte-identity.
HF_MIRROR_REPO = "naren-14/mvtec-ad-metal-nut"

# Published counts for metal_nut (Bergmann et al., IJCV 2021, Table 1).
EXPECTED_COUNTS = {
    "train/good": 220,
    "test/good": 22, "test/bent": 25, "test/color": 22, "test/flip": 23, "test/scratch": 23,
    "ground_truth/bent": 25, "ground_truth/color": 22, "ground_truth/flip": 23, "ground_truth/scratch": 23,
}


def download_hf_mirror(out_root: Path, repo: str = HF_MIRROR_REPO, timeout: float = 60.0) -> int:
    api = f"https://huggingface.co/api/datasets/{repo}"
    with urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": "insiso-downloader/1.0"}),
                                timeout=timeout) as r:
        info = json.load(r)
    rev = info["sha"]
    files = [s["rfilename"] for s in info["siblings"] if s["rfilename"].startswith(CATEGORY + "/")]
    root_resolved = out_root.resolve()
    log(f"mirror {repo}@{rev[:10]}: {len(files)} files")
    n = 0
    for i, rel in enumerate(files, 1):
        p = Path(rel)
        is_txt = rel in (f"{CATEGORY}/license.txt", f"{CATEGORY}/readme.txt")
        if p.is_absolute() or ".." in p.parts or not (rel.endswith(".png") or is_txt):
            raise RuntimeError(f"unsafe or unexpected mirror path refused: {rel}")
        dest = (out_root / rel).resolve()
        if root_resolved not in dest.parents:
            raise RuntimeError(f"path traversal blocked: {rel}")
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{urllib.request.quote(rel)}"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r:
                    data = r.read()
                if not is_txt and data[:8] != b"\x89PNG\r\n\x1a\n":
                    raise IOError(f"not a PNG: {rel}")
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(data)
                os.replace(tmp, dest)
                n += 1
                break
            except Exception as e:
                log(f"retry {rel}: {e}")
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"failed to fetch {rel}")
        if i % 50 == 0:
            log(f"{i}/{len(files)} files")
    (out_root / "MIRROR_REVISION.txt").write_text(f"{repo}@{rev}\n")
    return n


def count_check(cat_dir: Path) -> dict:
    """Compare per-folder PNG counts to the published numbers."""
    out = {}
    for rel, exp in EXPECTED_COUNTS.items():
        got = len(list((cat_dir / rel).glob("*.png")))
        out[rel] = {"expected": exp, "found": got, "ok": got == exp}
    return out


def verify_against_archive(cat_dir: Path, archive: Path) -> dict:
    """SHA-256 compare every PNG in the extracted tree with the official archive."""
    with tarfile.open(archive, "r:xz") as tf:
        ref = {}
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith(".png"):
                ref[m.name] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
    local = {f"{CATEGORY}/{p.relative_to(cat_dir).as_posix()}": sha256_of(p) for p in cat_dir.rglob("*.png")}
    return {
        "archive_files": len(ref), "local_files": len(local),
        "missing_locally": sorted(set(ref) - set(local)),
        "extra_locally": sorted(set(local) - set(ref)),
        "hash_mismatch": sorted(k for k in set(ref) & set(local) if ref[k] != local[k]),
    }


def write_provenance(out_root: Path, archive: Path | None, url: str, source: str) -> Path:
    prov = {
        "dataset": "MVTec Anomaly Detection (MVTec AD)",
        "category": CATEGORY,
        "source_url": url,
        "acquisition": source,
        "archive_name": archive.name if archive else None,
        "archive_bytes": archive.stat().st_size if archive and archive.exists() else None,
        "archive_sha256": sha256_of(archive) if archive and archive.exists() else None,
        "retrieved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": platform.platform(),
        "python": sys.version.split()[0],
        "licence": "CC BY-NC-SA 4.0 (non-commercial); see LICENSE_NOTES.md",
        "citation": "Bergmann et al., 'MVTec AD - A Comprehensive Real-World Dataset for "
                    "Unsupervised Anomaly Detection', CVPR 2019.",
    }
    p = out_root / "PROVENANCE.json"
    p.write_text(json.dumps(prov, indent=2))
    log(f"provenance written to {p}")
    return p


def ensure_dataset(data_root: Path, archive_dir: Path | None = None, url: str = METAL_NUT_URL,
                   force: bool = False, allow_mirror: bool = False) -> Path:
    """Idempotent entry point. Returns the path to the extracted `metal_nut` directory."""
    data_root = Path(data_root)
    archive_dir = Path(archive_dir) if archive_dir else data_root
    archive = archive_dir / ARCHIVE_NAME

    if dataset_is_complete(data_root) and not force:
        log(f"dataset already present at {data_root / CATEGORY}; skipping download")
        if not (data_root / "PROVENANCE.json").exists():
            write_provenance(data_root, archive if archive.exists() else None, url, "pre-existing directory")
        return data_root / CATEGORY

    source = "pre-placed archive"
    if not archive.exists():
        log(f"downloading {url}")
        try:
            download(url, archive)
            source = "automatic download"
        except Exception as e:
            log(f"automatic download failed: {e}")
            if allow_mirror:
                log(f"falling back to UNOFFICIAL mirror {HF_MIRROR_REPO} (--allow-mirror)")
                download_hf_mirror(data_root)
                checks = count_check(data_root / CATEGORY)
                for k, v in checks.items():
                    log(f"  {k:24s} expected {v['expected']:3d} found {v['found']:3d} {'OK' if v['ok'] else 'MISMATCH'}")
                if not all(v["ok"] for v in checks.values()):
                    raise RuntimeError("mirror contents do not match published MVTec counts")
                rev = (data_root / "MIRROR_REVISION.txt").read_text().strip()
                write_provenance(data_root, None, f"https://huggingface.co/datasets/{rev.split('@')[0]}",
                                 f"UNOFFICIAL Hugging Face mirror {rev}; official URL {url} returned: {e}")
                return data_root / CATEGORY
            log("MANUAL FALLBACK: download metal_nut.tar.xz from "
                "https://www.mvtec.com/company/research/datasets/mvtec-ad and place it at:")
            log(f"    {archive.resolve()}")
            log("then re-run this script.")
            raise SystemExit(2)
    else:
        log(f"using existing archive {archive}")

    validate_archive(archive)
    safe_extract(archive, data_root)
    if not dataset_is_complete(data_root):
        raise RuntimeError("extraction finished but expected train/test/ground_truth structure is missing")
    write_provenance(data_root, archive, url, source)
    return data_root / CATEGORY
