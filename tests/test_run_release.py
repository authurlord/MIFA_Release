from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_release", ROOT / "scripts/run_release.py")
assert SPEC is not None and SPEC.loader is not None
RUN_RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_RELEASE)


def test_selected_fused_cpu_kernels_pin_blas_to_one_thread() -> None:
    registry = RUN_RELEASE.load_kernel_registry()
    inherited = {
        "OMP_NUM_THREADS": "32",
        "NUMBA_NUM_THREADS": "32",
        "MKL_NUM_THREADS": "64",
        "OPENBLAS_NUM_THREADS": "64",
    }
    for dataset in ("qdrant_hm", "appreviews", "arxiv"):
        kernel = RUN_RELEASE.kernel_for(registry, dataset)
        env = RUN_RELEASE.environment_for(inherited, kernel, "cpu")
        assert env["OMP_NUM_THREADS"] == "32"
        assert env["NUMBA_NUM_THREADS"] == "32"
        assert env["MKL_NUM_THREADS"] == "1"
        assert env["OPENBLAS_NUM_THREADS"] == "1"


def test_gpu_only_and_unselected_cpu_kernels_preserve_existing_blas() -> None:
    registry = RUN_RELEASE.load_kernel_registry()
    inherited = {"MKL_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16"}
    selected = RUN_RELEASE.kernel_for(registry, "arxiv")
    legacy = RUN_RELEASE.kernel_for(registry, "wands")

    gpu_env = RUN_RELEASE.environment_for(inherited, selected, "1gpu")
    legacy_cpu_env = RUN_RELEASE.environment_for(inherited, legacy, "cpu")

    assert gpu_env["MKL_NUM_THREADS"] == "16"
    assert gpu_env["OPENBLAS_NUM_THREADS"] == "16"
    assert legacy_cpu_env["MKL_NUM_THREADS"] == "16"
    assert legacy_cpu_env["OPENBLAS_NUM_THREADS"] == "16"
