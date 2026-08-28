"""Tier-A imputers + the imputability probes used by Gate 1.

The critical one is EmbeddingKNNImputer: impute a missing scalar from the k
nearest neighbors in EMBEDDING space. Gate 1 compares its predictive power
against the marginal prior to decide whether MIFA can possibly beat drop-missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PriorBaseline:
    """Marginal prior: most-frequent (cat) / global mean (num)."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.value = None

    def fit(self, y):
        y = np.asarray(y)
        if self.dtype == "cat":
            vals, counts = np.unique(y, return_counts=True)
            self.value = vals[np.argmax(counts)]
        else:
            self.value = float(np.mean(y))
        return self

    def predict(self, n):
        return np.full(n, self.value)


def impute_table_embedding_knn(spec, observed_table, mask, cols, k=10):
    """Complete the masked table by embedding-kNN point imputation (done once).

    For each query-relevant attr, masked tuples are filled from the k nearest
    OBSERVED tuples of that attr in embedding space: majority vote (categorical)
    or mean (numerical). This is the 'first-impute-then-query' baseline.
    """
    from mifa.knn import knn_batch

    X = spec.normalized_embeddings()
    out = observed_table.copy()
    for j, c in enumerate(cols):
        col_missing = mask[:, j]
        if not col_missing.any():
            continue
        obs_idx = np.where(~col_missing)[0]
        miss_idx = np.where(col_missing)[0]
        y_obs = spec.table[c].to_numpy()[obs_idx]
        nn, _ = knn_batch(X[miss_idx], X[obs_idx], k)
        if out[c].dtype == bool or pd.api.types.is_integer_dtype(spec.table[c].dtype):
            out[c] = out[c].astype(object)
        if spec.dtypes[c] == "cat":
            filled = []
            for row in nn:
                vals, counts = np.unique(y_obs[row], return_counts=True)
                filled.append(vals[np.argmax(counts)])
            out.loc[miss_idx, c] = filled
        else:
            yk = y_obs[nn].astype(float)
            out.loc[miss_idx, c] = yk.mean(axis=1)
    return out


def _encode_features(table, feat_cols, dtypes, train_rows):
    """Build a numeric feature matrix from scalar columns for a tree imputer.

    num: value (NaN->train median) + missing indicator. cat: category codes
    (NaN->-1) + missing indicator. Fill stats come from train_rows only (no leakage).
    """
    blocks = []
    for c in feat_cols:
        col = table[c]
        miss = col.isna().to_numpy()
        if dtypes[c] == "cat":
            codes = col.astype("category").cat.codes.to_numpy().astype(float)  # NaN -> -1
            blocks.append(codes[:, None])
        else:
            vals = col.to_numpy().astype(float)
            med = np.nanmedian(vals[train_rows]) if train_rows.any() else 0.0
            vals = np.where(np.isnan(vals), med if np.isfinite(med) else 0.0, vals)
            blocks.append(vals[:, None])
        blocks.append(miss.astype(float)[:, None])
    return np.hstack(blocks) if blocks else np.zeros((len(table), 0))


def impute_table_dependency(spec, observed_table, mask, cols, n_estimators=100,
                            max_depth=None, seed=0, n_jobs=16):
    """Dependency imputer (MissForest-style): fill each missing query-relevant attr
    from the OTHER scalar attributes via a RandomForest trained only on rows where the
    target is observed. This is the fair 'relational first-impute-then-search' baseline
    for the quality half (R2Q) -- it uses inter-attribute dependencies like rule-MIFA,
    but as an off-the-shelf model rather than discovered Sigma_ind/Sigma_dep + I_mv.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    out = observed_table.copy()
    all_cols = list(spec.dtypes.keys())
    for j, c in enumerate(cols):
        col_missing = mask[:, j]
        if not col_missing.any():
            continue
        feat_cols = [fc for fc in all_cols if fc != c]
        obs_rows = ~col_missing                      # target observed (train)
        miss_idx = np.where(col_missing)[0]
        F = _encode_features(observed_table, feat_cols, spec.dtypes, obs_rows)
        Xtr, Xte = F[obs_rows], F[miss_idx]
        ytr = spec.table[c].to_numpy()[obs_rows]
        if spec.dtypes[c] == "cat":
            out[c] = out[c].astype(object)            # hold predicted labels of any dtype
            m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=seed, n_jobs=n_jobs)
            m.fit(Xtr, ytr)                           # raw labels (float/int/str) as classes
            out.loc[miss_idx, c] = m.predict(Xte)
        else:
            out[c] = out[c].astype(np.float64)        # avoid float64->float32 loc coercion error
            m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=seed, n_jobs=n_jobs)
            m.fit(Xtr, ytr.astype(float))
            out.loc[miss_idx, c] = m.predict(Xte).astype(float)
    return out


class EmbeddingKNNImputer:
    """Predict a target attr from embedding-space neighbors.

    For categorical: majority vote among k neighbors.
    For numerical:   mean of k neighbors (+ [Q25,Q75] interval for interval-hit).
    """

    def __init__(self, dtype, k=10):
        self.dtype = dtype
        self.k = k

    def predict_holdout(self, X_train, y_train, X_query, observed_mask=None):
        """For each query point, predict from k nearest TRAIN neighbors.

        Uses batched BLAS/GPU k-NN (mifa.knn) instead of a per-query Python loop.
        If observed_mask is given (bool over train, True=value observed), only
        observed neighbors vote; if none of the retrieved neighbors are observed,
        the prediction abstains (None) -> simulates the real incomplete database.
        To keep enough observed votes under masking, we retrieve a wider pool.
        """
        from mifa.knn import knn_batch

        y_train = np.asarray(y_train)
        # retrieve a wider pool when masking so ~k observed neighbors remain
        pool = self.k if observed_mask is None else min(len(y_train), self.k * 4)
        nn_ids, _ = knn_batch(X_query, X_train, pool)

        preds = []
        intervals = []
        for row in nn_ids:
            nn = row
            if observed_mask is not None:
                nn = nn[observed_mask[nn]][: self.k]
            else:
                nn = nn[: self.k]
            if len(nn) == 0:
                preds.append(None)
                intervals.append((None, None))
                continue
            yv = y_train[nn]
            if self.dtype == "cat":
                vals, counts = np.unique(yv, return_counts=True)
                preds.append(vals[np.argmax(counts)])
                intervals.append((None, None))
            else:
                yvf = yv.astype(float)
                preds.append(float(np.mean(yvf)))
                intervals.append((float(np.quantile(yvf, 0.25)),
                                  float(np.quantile(yvf, 0.75))))
        return np.array(preds, dtype=object), intervals
