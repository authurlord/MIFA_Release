"""Transport adapter for official R MICE and MissForest release packages."""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RSCRIPT = (
    ROOT / ".local" / "mifa_ext" / "r-paper-baselines" / "bin" / "Rscript")
ADAPTER = ROOT / "scripts" / "release_imputation_adapter.R"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ReleaseMaterialization:
    table: pd.DataFrame
    method: str
    package_version: str
    protocol: str
    construction_seconds: float
    train_rows: int
    target_rows: int
    adapter_sha256: str


def materialize_release_imputer(
    method: str,
    train: pd.DataFrame,
    target: pd.DataFrame,
    dtypes: dict[str, str],
    *,
    seed: int,
    max_iter: int = 5,
    n_trees: int = 50,
    threads: int = 8,
    rscript: Path = DEFAULT_RSCRIPT,
) -> ReleaseMaterialization:
    """Run an official release package without reimplementing its algorithm."""
    if method not in {"mice", "missforest"}:
        raise ValueError(f"unsupported release imputer: {method}")
    if list(train.columns) != list(target.columns):
        raise ValueError("train and target columns must match exactly")
    unknown = set(train.columns) - set(dtypes)
    if unknown:
        raise ValueError(f"missing dtype declarations: {sorted(unknown)}")
    if not rscript.is_file():
        raise FileNotFoundError(rscript)

    with tempfile.TemporaryDirectory(prefix=f"mifa_release_{method}_") as tmp:
        tmp = Path(tmp)
        train_path = tmp / "train.csv"
        target_path = tmp / "target.csv"
        schema_path = tmp / "schema.tsv"
        output_path = tmp / "completed.csv"
        train.to_csv(train_path, index=False, na_rep="")
        target.to_csv(target_path, index=False, na_rep="")
        schema_path.write_text("".join(
            f"{column}\t{dtypes[column]}\n" for column in train.columns))
        env = os.environ.copy()
        # The paper releases expose sequential fitting APIs.  Pin underlying
        # numerical libraries to one thread rather than adding baseline-side
        # parallelism; ``threads`` remains only the experiment-wide ceiling.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        result = subprocess.run([
            str(rscript), str(ADAPTER), method, str(train_path),
            str(target_path), str(schema_path), str(output_path), str(seed),
            str(max_iter), str(n_trees), str(threads),
        ], check=False, capture_output=True, text=True, env=env)
        if result.returncode:
            raise RuntimeError(
                f"official {method} release failed: {result.stderr[-2000:]}")
        meta_lines = [line for line in result.stdout.splitlines()
                      if line.startswith("MIFA_RELEASE_IMPUTER_META\t")]
        if len(meta_lines) != 1:
            raise RuntimeError(f"missing release metadata: {result.stdout[-2000:]}")
        _, got_method, version, protocol, seconds, train_rows, target_rows = \
            meta_lines[0].split("\t")
        completed = pd.read_csv(output_path, keep_default_na=False, na_values=[""])

    completed = completed[list(target.columns)]
    categorical_numeric: dict[str, bool] = {}
    for column, kind in dtypes.items():
        if kind == "num":
            completed[column] = pd.to_numeric(completed[column], errors="raise")
        else:
            # ``cat`` describes predicate semantics, not necessarily the
            # physical scalar type.  MovieLens ``decade``, for example, is a
            # categorical attribute whose query literals are numeric.  Keep
            # that representation so equality predicates see the same values
            # before and after the R CSV transport.
            categorical_numeric[column] = bool(
                pd.api.types.is_numeric_dtype(train[column].dtype)
                or pd.api.types.is_numeric_dtype(target[column].dtype)
            )
            if categorical_numeric[column]:
                completed[column] = pd.to_numeric(
                    completed[column], errors="raise")
            else:
                completed[column] = completed[column].astype("string")
    if len(completed) != len(target):
        raise RuntimeError("official release changed target row count")
    # Imputation must not modify cells that were already observed.
    for column in target.columns:
        observed = target[column].notna()
        if dtypes[column] == "num":
            left = pd.to_numeric(target.loc[observed, column]).astype(float)
            right = completed.loc[observed, column].astype(float)
            # The transport crosses float32 parquet -> decimal CSV -> R double
            # -> decimal CSV.  Preserve observed cells up to the source
            # float32 representation instead of requiring bit equality after
            # the decimal round trip.
            if not np.allclose(
                    left.to_numpy(), right.to_numpy(),
                    rtol=5e-7, atol=5e-7, equal_nan=False):
                raise RuntimeError(f"{method} changed observed numeric cells: {column}")
        elif categorical_numeric.get(column, False):
            left = pd.to_numeric(target.loc[observed, column]).astype(float)
            right = pd.to_numeric(completed.loc[observed, column]).astype(float)
            if not np.allclose(
                    left.to_numpy(), right.to_numpy(),
                    rtol=5e-7, atol=5e-7, equal_nan=False):
                raise RuntimeError(
                    f"{method} changed observed categorical-numeric cells: "
                    f"{column}")
        else:
            left = target.loc[observed, column].astype(str).to_numpy()
            right = completed.loc[observed, column].astype(str).to_numpy()
            if not (left == right).all():
                raise RuntimeError(f"{method} changed observed categorical cells: {column}")
    if completed.isna().any(axis=None):
        missing = completed.columns[completed.isna().any()].tolist()
        raise RuntimeError(f"{method} left missing cells: {missing}")
    return ReleaseMaterialization(
        table=completed,
        method=got_method,
        package_version=version,
        protocol=protocol,
        construction_seconds=float(seconds),
        train_rows=int(train_rows),
        target_rows=int(target_rows),
        adapter_sha256=_sha256(ADAPTER),
    )
