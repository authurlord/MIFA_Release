#!/usr/bin/env python3
"""Reproduce the paper's two-GPU process-replica deployment.

Each GPU owns one independent serving process and the complete corpus.  P50
and P95 use the slower replica; throughput is the sum of concurrent saturated
single-GPU throughput.  This is distinct from corpus sharding inside one
process.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/release_experiments.json"
RUNNER = ROOT / "scripts/run_release.py"


def datasets(value: str) -> list[dict]:
    rows = json.loads(CONFIG.read_text(encoding="utf-8"))["datasets"]
    if value == "all":
        return rows
    wanted = {item.strip() for item in value.split(",") if item.strip()}
    selected = [row for row in rows if row["key"] in wanted]
    missing = wanted - {row["key"] for row in selected}
    if missing:
        raise ValueError(f"unknown dataset keys: {sorted(missing)}")
    return selected


def report_path(row: dict, mechanism: str, replica: int) -> Path:
    tag = f"release_replica{replica}_{row['key']}_{mechanism}_1gpu"
    return ROOT / "reports" / (
        f"paper_aligned_static_{row['key']}_seed0_{tag}.json")


def one_gpu_metrics(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = (report.get("candidate_efficiency") or {}).get("rows") or []
    row = next(item for item in rows if int(item["gpu_count"]) == 1)
    response = row["response_batch1"]
    throughput = row.get("response_batch_throughput") or {}
    return {
        "f1_at_10": float(report["results"]["mifa_paper_static"]["macro"]["f1"]),
        "p50_ms": float(response["latency_ms_p50"]),
        "p95_ms": float(response["latency_ms_p95"]),
        "qps": float(throughput.get(
            "aggregate_qps_median", response["qps_from_mean"])),
        "resident_bytes": int((row.get("provider") or {}).get("resident_bytes") or 0),
        "peak_gpu_bytes": int(response.get("peak_gpu_allocated_bytes") or 0),
    }


def run_dataset(row: dict, mechanism: str, injected_root: Path,
                physical_devices: tuple[str, str], output: Path) -> dict:
    processes = []
    streams = []
    for replica, device in enumerate(physical_devices):
        log = output / row["key"] / f"replica{replica}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("w")
        command = [
            sys.executable, str(RUNNER),
            "--datasets", row["key"],
            "--mechanism", mechanism,
            "--platforms", "1gpu",
            "--injected-root", str(injected_root),
            "--tag", f"release_replica{replica}",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = device
        env.setdefault("OMP_NUM_THREADS", "16")
        env.setdefault("NUMBA_NUM_THREADS", "16")
        env.setdefault("MKL_NUM_THREADS", "16")
        env.setdefault("OPENBLAS_NUM_THREADS", "16")
        processes.append(subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT))
        streams.append(stream)
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        raise RuntimeError(f"{row['key']} replica return codes: {codes}")
    left = one_gpu_metrics(report_path(row, mechanism, 0))
    right = one_gpu_metrics(report_path(row, mechanism, 1))
    return {
        "dataset": row["display"],
        "dataset_key": row["key"],
        "mechanism": mechanism,
        "platform": "2-GPU process replicas",
        "f1_at_10": min(left["f1_at_10"], right["f1_at_10"]),
        "p50_ms": max(left["p50_ms"], right["p50_ms"]),
        "p95_ms": max(left["p95_ms"], right["p95_ms"]),
        "qps": left["qps"] + right["qps"],
        "replica0_qps": left["qps"],
        "replica1_qps": right["qps"],
        "resident_bytes": left["resident_bytes"] + right["resident_bytes"],
        "peak_gpu_bytes": left["peak_gpu_bytes"] + right["peak_gpu_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--mechanism", choices=("mcar", "hce", "qae"),
                        default="hce")
    parser.add_argument("--injected-root", type=Path,
                        default=Path("data/injected/hce"))
    parser.add_argument("--physical-devices", default="0,1")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/two_gpu_process_replicas"))
    args = parser.parse_args()
    devices = tuple(item.strip() for item in args.physical_devices.split(","))
    if len(devices) != 2:
        parser.error("--physical-devices requires exactly two GPU IDs")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    rows = [run_dataset(
        row, args.mechanism, args.injected_root.resolve(),
        (devices[0], devices[1]), output)
        for row in datasets(args.datasets)]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "MIFA_TWO_GPU_PROCESS_REPLICAS.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

