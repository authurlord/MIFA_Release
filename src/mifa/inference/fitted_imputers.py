"""Train-only fit/transform imputers for the leakage-resistant v2 harness."""
from __future__ import annotations

import numpy as np
import pandas as pd


class MeanModeImputer:
    def __init__(self, dtypes):
        self.dtypes = dtypes
        self.values = {}

    def fit(self, table):
        for col, dtype in self.dtypes.items():
            observed = table[col].dropna()
            if len(observed) == 0:
                self.values[col] = None
            elif dtype == "cat":
                self.values[col] = observed.mode(dropna=True).iloc[0]
            else:
                self.values[col] = float(observed.astype(float).mean())
        return self

    def transform(self, table):
        out = table.copy()
        for col, value in self.values.items():
            if value is None:
                continue
            if self.dtypes[col] == "cat":
                out[col] = out[col].astype(object)
            else:
                out[col] = out[col].astype(float)
            out[col] = out[col].fillna(value)
        return out


class MixedIterativeImputer:
    """MICE/MissForest over a training-fitted mixed-table numeric encoding."""

    def __init__(self, dtypes, *, kind="mice", seed=0, max_iter=5,
                 n_estimators=50, n_jobs=8):
        if kind not in {"mice", "missforest"}:
            raise ValueError("kind must be mice or missforest")
        self.dtypes = dict(dtypes)
        self.columns = list(dtypes)
        self.kind = kind
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.n_estimators = int(n_estimators)
        self.n_jobs = int(n_jobs)
        self.categories = {}

    def _encode(self, table, *, fitting):
        work = np.full((len(table), len(self.columns)), np.nan, dtype=float)
        for j, col in enumerate(self.columns):
            series = table[col]
            if self.dtypes[col] == "num":
                work[:, j] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                continue
            if fitting:
                self.categories[col] = list(pd.unique(series.dropna()))
            mapping = {value: i for i, value in enumerate(self.categories[col])}
            values = np.full(len(series), np.nan, dtype=float)
            observed = ~series.isna().to_numpy()
            # Unknown observed categories stay at -1 as a feature value; genuinely
            # missing cells remain NaN and are imputed from the training vocabulary.
            values[observed] = [mapping.get(value, -1) for value in series[observed]]
            work[:, j] = values
        return work

    def fit(self, table):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        if self.kind == "mice":
            from sklearn.linear_model import BayesianRidge
            estimator = BayesianRidge()
        else:
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.seed,
                n_jobs=self.n_jobs,
            )
        self.model = IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            random_state=self.seed,
            sample_posterior=False,
            initial_strategy="most_frequent",
            keep_empty_features=True,
            skip_complete=False,
        )
        self.model.fit(self._encode(table, fitting=True))
        return self

    def transform(self, table):
        original = self._encode(table, fitting=False)
        filled = self.model.transform(original)
        out = table.copy()
        for j, col in enumerate(self.columns):
            missing = table[col].isna().to_numpy()
            if not missing.any():
                continue
            if self.dtypes[col] == "num":
                out[col] = out[col].astype(float)
                out.loc[missing, col] = filled[missing, j]
                continue
            cats = self.categories[col]
            if not cats:
                continue
            codes = np.clip(np.rint(filled[missing, j]).astype(int), 0, len(cats) - 1)
            out[col] = out[col].astype(object)
            out.loc[missing, col] = [cats[code] for code in codes]
        return out
