#!/usr/bin/env python3
"""
process.py — Underwater image enhancement across all algorithms.

Usage:
    python process.py <input_dir> [-j N]

    input_dir   folder of images, e.g. C60/
    -j N        parallel worker processes (default: all CPU cores)
    -j 1        sequential mode (useful for debugging)

Example:
    python process.py C60/ -j 8

    C60/xxx.jpg  →  C60_CLAHE/xxx.jpg
                    C60_GC/xxx.jpg
                    C60_ICM/xxx.jpg
                    C60_RGHS/xxx.jpg
                    C60_UCM/xxx.jpg

Place this file next to the CLAHE/, GC/, ICM/, RGHS/, UCM/ folders.
"""

import os
import sys
import cv2
import numpy as np
import argparse
import warnings
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# ── directory that contains CLAHE/, GC/, ICM/, RGHS/, UCM/ ──────────────────
BASE_DIR = Path(__file__).resolve().parent

ALGO_DIRS = {
    "CLAHE": BASE_DIR / "CLAHE",
    "GC":    BASE_DIR / "GC",
    "ICM":   BASE_DIR / "ICM",
    "RGHS":  BASE_DIR / "RGHS",
    "UCM":   BASE_DIR / "UCM",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── temporarily expose one algorithm folder to the import system ─────────────

@contextmanager
def _algo_scope(algo_dir: Path):
    path_str = str(algo_dir.resolve())
    sys.path.insert(0, path_str)
    existing = set(sys.modules)
    try:
        yield
    finally:
        if path_str in sys.path:
            sys.path.remove(path_str)
        for key in list(sys.modules):
            if key in existing:
                continue
            mod = sys.modules.get(key)
            if mod is None:
                continue
            mod_file = getattr(mod, "__file__", None) or ""
            resolved = str(Path(mod_file).resolve()) if mod_file else ""
            if resolved.startswith(path_str + os.sep) or resolved == path_str:
                del sys.modules[key]


# ── per-algorithm pipelines ──────────────────────────────────────────────────

def _apply_CLAHE(img: np.ndarray) -> np.ndarray:
    with _algo_scope(ALGO_DIRS["CLAHE"]):
        from sceneRadianceCLAHE import RecoverCLAHE
        return RecoverCLAHE(img.copy())


def _apply_GC(img: np.ndarray) -> np.ndarray:
    with _algo_scope(ALGO_DIRS["GC"]):
        from sceneRadianceGC import RecoverGC
        return RecoverGC(img.copy())


def _apply_ICM(img: np.ndarray) -> np.ndarray:
    with _algo_scope(ALGO_DIRS["ICM"]):
        from global_histogram_stretching import stretching
        from hsvStretching import HSVStretching
        from sceneRadiance import sceneRadianceRGB
        out = stretching(img.copy())
        out = sceneRadianceRGB(out)
        out = HSVStretching(out)
        out = sceneRadianceRGB(out)
        return out


def _apply_RGHS(img: np.ndarray) -> np.ndarray:
    with _algo_scope(ALGO_DIRS["RGHS"]):
        from global_stretching_RGB import stretching
        from LabStretching import LABStretching
        out = stretching(img.copy())
        out = LABStretching(out)
        return np.clip(out, 0, 255).astype(np.uint8)


def _apply_UCM(img: np.ndarray) -> np.ndarray:
    with _algo_scope(ALGO_DIRS["UCM"]):
        from color_equalisation import RGB_equalisation
        from global_histogram_stretching import stretching
        from hsvStretching import HSVStretching
        from sceneRadiance import sceneRadianceRGB
        out = RGB_equalisation(img.copy())
        out = stretching(out)
        out = HSVStretching(out)
        out = sceneRadianceRGB(out)
        return out


ALGORITHMS = {
    "CLAHE": _apply_CLAHE,
    "GC":    _apply_GC,
    "ICM":   _apply_ICM,
    "UCM":   _apply_UCM,
}


# ── top-level worker (must be module-level for pickle) ───────────────────────

def _process_one(img_path_str: str, out_path_str: str, algo_name: str) -> tuple:
    """Read → enhance → write. Returns ('ok'|'skip'|'err', message|None)."""
    warnings.filterwarnings("ignore")
    img = cv2.imread(img_path_str)
    if img is None:
        return ("skip", Path(img_path_str).name)
    try:
        result = ALGORITHMS[algo_name](img)
        cv2.imwrite(out_path_str, result)
        return ("ok", None)
    except Exception as exc:
        return ("err", str(exc))


# ── main ─────────────────────────────────────────────────────────────────────

def process(input_dir: Path, output_dir: Path, num_workers: int) -> None:
    input_dir  = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        sys.exit(f"[ERROR] Not a directory: {input_dir}")

    images = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    )
    if not images:
        sys.exit(f"[ERROR] No images found in {input_dir}")

    # build output dirs: <output_dir>/<input_name>_<ALGO>/
    out_dirs = {}
    for algo_name in ALGORITHMS:
        d = output_dir / f"{input_dir.name}_{algo_name}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[algo_name] = d

    # build task list: every (image × algorithm) pair
    tasks = [
        (str(img), str(out_dirs[algo] / img.name), algo)
        for img in images
        for algo in ALGORITHMS
    ]

    total = len(tasks)
    effective_workers = min(num_workers, total)
    print(f"{input_dir.name}/  —  {len(images)} image(s) × {len(ALGORITHMS)} algorithms "
          f"= {total} tasks  |  workers: {effective_workers}\n")

    ok_counts  = defaultdict(int)
    err_counts = defaultdict(int)
    done = 0

    def _tick(img_str, algo, status, msg):
        nonlocal done
        folder = Path(img_str).parent.name
        if status == "ok":
            ok_counts[algo] += 1
        elif status == "err":
            err_counts[algo] += 1
            print(f"\n  [ERR] {algo} ← {Path(img_str).name}: {msg}", flush=True)
        done += 1
        print(f"\r  {done}/{total}", end="", flush=True)

    if effective_workers == 1:
        warnings.filterwarnings("ignore")
        for img_str, out_str, algo in tasks:
            status, msg = _process_one(img_str, out_str, algo)
            _tick(img_str, algo, status, msg)
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            future_map = {
                pool.submit(_process_one, img_str, out_str, algo): (img_str, algo)
                for img_str, out_str, algo in tasks
            }
            for future in as_completed(future_map):
                img_str, algo = future_map[future]
                try:
                    status, msg = future.result()
                except Exception as exc:
                    status, msg = "err", str(exc)
                _tick(img_str, algo, status, msg)

    print("\n")
    for algo_name in ALGORITHMS:
        ok  = ok_counts[algo_name]
        err = err_counts[algo_name]
        status = f"{ok} saved" + (f", {err} error(s)" if err else "")
        print(f"  {algo_name:<6} → {output_dir / f'{input_dir.name}_{algo_name}'}  [{status}]")

    print(f"\n✓ Done — {sum(ok_counts.values())} enhanced images written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch underwater image enhancement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", type=Path,
                        help="Folder of input images (e.g. C60/)")
    parser.add_argument("output_dir", type=Path,
                        help="Root folder for output (e.g. results/); "
                             "sub-folders like C60_CLAHE/ are created inside it")
    parser.add_argument("-j", "--workers", type=int, default=os.cpu_count(),
                        metavar="N",
                        help=f"parallel workers (default: {os.cpu_count()})")
    args = parser.parse_args()
    process(args.input_dir, args.output_dir, args.workers)
