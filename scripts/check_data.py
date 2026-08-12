"""
Validates the DARES HDF5 containers against the loader contract.

For every split this script reports the dataset keys, shapes, dtypes,
compression, per-split forest ratio (validating the [15%, 85%] patch-balancing
protocol from ``Docs/data.md``) and the effective batch count per epoch.

Examples:
    python scripts/check_data.py --source_dir data/source --target_dir data/target
    python scripts/check_data.py --config configs/training/dares.yaml --batch_size 8
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from dares.config import ExperimentConfig
from dares.data.loader import SPLIT_FILENAMES

FOREST_RATIO_RANGE = (0.15, 0.85)


def _check_container(path: Path, split: str, batch_size: int) -> dict:
    """Checks a single HDF5 container and returns a summary dict."""
    with h5py.File(path, "r") as f:
        images = f["images"]
        n, c, h, w = images.shape
        info = {
            "path": path.name,
            "split": split,
            "n_patches": n,
            "shape": f"{c}x{h}x{w}",
            "dtype": str(images.dtype),
            "compression": str(images.compression),
        }

        mask = f.get("masks")
        if mask is not None:
            info["mask_dtype"] = str(mask.dtype)
            mask_vals = mask[...]
            unique = np.unique(mask_vals)
            info["mask_values"] = [int(v) for v in unique]
            ratios = mask_vals.mean(axis=(1, 2))
            info["forest_ratio"] = float(ratios.mean())
            info["ratio_min"] = float(ratios.min())
            info["ratio_max"] = float(ratios.max())
            info["out_of_range"] = int(
                ((ratios < FOREST_RATIO_RANGE[0]) | (ratios > FOREST_RATIO_RANGE[1])).sum()
            )
        else:
            info["mask_dtype"] = None
            info["forest_ratio"] = None

        info["effective_batches"] = (
            n // batch_size if split == "train" else (n + batch_size - 1) // batch_size
        )
        return info


def main(
    source_dir: str,
    target_dir: str,
    batch_size: int = 8,
) -> None:
    """Validates all six containers and prints a summary table.

    Args:
        source_dir (str): Directory containing ``source_*.h5``.
        target_dir (str): Directory containing ``target_*.h5``.
        batch_size (int): Training batch size (affects the batch-count report).
    """
    header = (
        f"{'file':<22}{'split':<11}{'N':>6}{'shape':>12}{'dtype':>9}"
        f"{'comp':>8}{'mask':>8}{'forest%':>9}"
    )
    print("=" * len(header))
    print("DARES HDF5 container validation")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    summary = []
    for domain, domain_dir in (("source", source_dir), ("target", target_dir)):
        domain_path = Path(domain_dir)
        if not domain_path.is_dir():
            raise FileNotFoundError(f"domain directory not found: {domain_path}")
        for split, filename in SPLIT_FILENAMES.items():
            path = domain_path / filename.format(domain)
            if not path.is_file():
                raise FileNotFoundError(f"split container not found: {path}")
            info = _check_container(path, split, batch_size)
            summary.append(info)
            forest = f"{info['forest_ratio']:.3f}" if info["forest_ratio"] is not None else "-"
            print(
                f"{info['path']:<22}{info['split']:<11}{info['n_patches']:>6}"
                f"{info['shape']:>12}{info['dtype']:>9}{info['compression']:>8}"
                f"{info['mask_dtype'] or '-':>8}{forest:>9}"
            )

    print("-" * len(header))
    for info in summary:
        ratio = info["forest_ratio"]
        if ratio is not None:
            print(
                f"{info['path']}: forest ratio {ratio:.3f} "
                f"[{info['ratio_min']:.3f}, {info['ratio_max']:.3f}], "
                f"{info['out_of_range']} patch(es) outside "
                f"{FOREST_RATIO_RANGE}, {info['effective_batches']} batches/epoch"
            )
        else:
            print(
                f"{info['path']}: unlabeled (no masks), "
                f"{info['effective_batches']} batches/epoch"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DARES HDF5 data validator")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a DARES YAML config (resolves source/target dirs and batch size)",
    )
    parser.add_argument(
        "--source_dir", type=str, default=None, help="Source domain directory"
    )
    parser.add_argument(
        "--target_dir", type=str, default=None, help="Target domain directory"
    )
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    if args.config is not None:
        cfg = ExperimentConfig.from_yaml(args.config)
        source_dir = str(cfg.data.resolved_dir("source_dir"))
        target_dir = str(cfg.data.resolved_dir("target_dir"))
        batch_size = cfg.data.batch_size if args.batch_size is None else args.batch_size
        print(f"Using data dirs from {args.config}")
        print(f"  source_dir = {source_dir}")
        print(f"  target_dir = {target_dir}")
        main(source_dir, target_dir, batch_size)
    else:
        if args.source_dir is None or args.target_dir is None:
            raise SystemExit(
                "provide --config OR both --source_dir and --target_dir"
            )
        main(args.source_dir, args.target_dir, args.batch_size or 8)
