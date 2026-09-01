#!/usr/bin/env python3
"""One entry point for paper-aligned CPU and GPU MIFA runs.

The command is single-seed and exact.  It automatically loads dataset-specific
serving choices from ``configs/kernel_registry.json`` so each workload keeps
its selected CPU and GPU implementation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/release_experiments.json"
KERNELS = ROOT / "configs/kernel_registry.json"
RUNNER = ROOT / "scripts/experiment_paper_aligned.py"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_kernel_registry() -> dict:
    registry = load_json(KERNELS)
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported kernel registry schema")
    keys = [row.get("dataset_key") for row in registry.get("datasets", [])]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("kernel registry requires unique dataset keys")
    return registry


def selected_datasets(config: dict, value: str) -> list[dict]:
    if value == "all":
        return list(config["datasets"])
    wanted = {item.strip() for item in value.split(",") if item.strip()}
    rows = [row for row in config["datasets"] if row["key"] in wanted]
    missing = wanted - {row["key"] for row in rows}
    if missing:
        raise ValueError(f"unknown dataset keys: {sorted(missing)}")
    return rows


def kernel_for(registry: dict, key: str) -> dict:
    for row in registry["datasets"]:
        if row["dataset_key"] == key:
            return row
    raise ValueError(f"no release kernel registered for {key}")


def command_for(row: dict, kernel: dict, mechanism: str, platforms: str,
                tag: str, export_only: bool,
                injection_seed: int | None = None) -> list[str]:
    validation, calibration, test = row["queries"]
    projection, leaf = row["selected"]["structure"].split("_leaf", 1)
    policy = row["selected"]["policy"]
    policy_grid = re.search(r"_s(\d+)_k(\d+)$", policy)
    if policy_grid is None:
        raise ValueError(f"cannot derive support/shrinkage grid from {policy}")
    support, shrinkage = policy_grid.groups()
    page_initial = (
        kernel["gpu_page_initial"]
        if platforms in {"1gpu", "2gpu"}
        else kernel["cpu_page_initial"])
    command = [
        sys.executable, str(RUNNER),
        "--dataset-config", row["dataset_config"],
        "--paper-config", row["paper_config"],
        "--seed", "0",
        "--rho", "0.20",
        "--missingness-kind", mechanism,
        "--historical-beta-exp", "0.50",
        "--historical-topk-multiplier", "10",
        "--answer-impact-fraction", "1.0",
        "--answer-cells-per-tuple", "1",
        "--n-validation-queries", str(validation),
        "--n-calibration-queries", str(calibration),
        "--n-test-queries", str(test),
        "--workload-profile", "canonical",
        "--k", "10",
        "--projection-dims", projection.removeprefix("p"),
        "--leaf-capacities", leaf,
        "--support-grid", support,
        "--shrinkage-grid", shrinkage,
        "--policy-families", "global,location,scalar,full",
        "--deployed-policy-candidate", policy,
        "--deployed-theta", str(row["selected"]["theta"]),
        "--candidate-workers", "4",
        "--workload-reference", "complete_train",
        "--ann-backend", "exact",
        "--leaf-frontier",
        "--executor-path", "unified_tree",
        "--skip-predicate-diagnostics",
        "--tag", tag,
    ]
    if mechanism != "tuple_mcar":
        command.extend(("--block-attrs", row["block_attrs"]))
    if injection_seed is not None:
        command.extend(("--injection-seed", str(injection_seed)))
    if row["embedding"]:
        command.extend(("--emb", row["embedding"]))
    if not export_only:
        command.extend([
            "--candidate-efficiency-matrix",
            "--candidate-efficiency-query-count", str(test),
            "--candidate-efficiency-repeats", "1",
            "--candidate-efficiency-platforms", platforms,
            "--candidate-efficiency-gpu-devices", "cuda:0,cuda:1",
            "--candidate-page-initial", str(page_initial),
            "--candidate-page-maximum", "65536",
            "--candidate-efficiency-cpu-projection-dim", "16",
            "--candidate-efficiency-cpu-refine-width", "64",
            "--candidate-efficiency-cpu-fused-exact-min-work",
            str(kernel["cpu_fused_exact_min_work"]),
            "--candidate-efficiency-exact-prewarm-strategy",
            kernel["exact_prewarm_strategy"],
            "--candidate-efficiency-response-only",
            "--candidate-efficiency-response-saturated-throughput",
            "--candidate-efficiency-response-use-frozen-reference",
        ])
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--mechanism", choices=("mcar", "hce", "qae"),
                        default="hce")
    parser.add_argument("--platforms", choices=("cpu", "1gpu", "2gpu", "all"),
                        default="cpu")
    parser.add_argument("--injected-root", type=Path)
    parser.add_argument("--export-injected-root", type=Path)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--tag", default="release")
    parser.add_argument(
        "--injection-seed", type=int,
        help=("vary only injected missingness while retaining the frozen "
              "seed-0 split, query cohort, deployment, and kernel"))
    args = parser.parse_args()

    config = load_json(MATRIX)
    registry = load_kernel_registry()
    mechanism = config["mechanisms"][args.mechanism]
    platforms = "cpu,1gpu,2gpu" if args.platforms == "all" else args.platforms
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "32")
    env.setdefault("NUMBA_NUM_THREADS", "32")
    env.setdefault("MKL_NUM_THREADS", "32")
    env.setdefault("OPENBLAS_NUM_THREADS", "32")
    if args.injected_root:
        env["MIFA_INJECTED_DATA_ROOT"] = str(args.injected_root.resolve())
    if args.export_injected_root:
        env["MIFA_EXPORT_INJECTED_ROOT"] = str(
            args.export_injected_root.resolve())
        env["MIFA_INJECTION_LABEL"] = args.mechanism
    if args.export_only:
        if not args.export_injected_root:
            parser.error("--export-only requires --export-injected-root")
        env["MIFA_EXPORT_INJECTED_ONLY"] = "1"

    for row in selected_datasets(config, args.datasets):
        kernel = kernel_for(registry, row["key"])
        tag = f"{args.tag}_{row['key']}_{args.mechanism}_{args.platforms}"
        command = command_for(
            row, kernel, mechanism, platforms, tag, args.export_only,
            args.injection_seed)
        print(json.dumps({
            "dataset": row["key"],
            "mechanism": args.mechanism,
            "platforms": platforms,
            "kernel_registry": str(KERNELS.relative_to(ROOT)),
            "kernel": {
                "cpu": kernel["cpu_kernel"],
                "gpu": kernel["gpu_kernel"],
            },
            "command": command,
        }), flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
