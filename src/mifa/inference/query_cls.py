"""Training-only query-conditioned classifier baseline for paper protocol v2.

The model is trained on pseudo-missing tuple--predicate pairs.  Hidden values
provide labels only; feature construction sees the masked scalar row, its mask,
a training-fitted embedding projection, and a fixed predicate encoding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mifa.predicates import TRUE, UNKNOWN


_OP_CODE = {"eq": 0.0, "in": 0.2, "le": 0.4, "ge": 0.6, "between": 0.8}


class QueryConditionedClassifier:
    """Predict joint residual predicate satisfaction for unresolved tuples."""

    def __init__(self, dtypes, projector, *, seed=0, max_iter=120,
                 max_train_pairs=120_000, max_predict_pairs=250_000):
        self.dtypes = dict(dtypes)
        self.attrs = list(dtypes)
        self.projector = projector
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.max_train_pairs = int(max_train_pairs)
        self.max_predict_pairs = int(max_predict_pairs)
        self.category_maps = {}
        self.category_freq = {}
        self.numeric_stats = {}
        self.model = None
        self.manifest = {}

    def _fit_scalar_encoding(self, observed):
        for attr in self.attrs:
            values = observed[attr].dropna()
            if self.dtypes[attr] == "num":
                numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
                center = float(np.mean(numeric)) if len(numeric) else 0.0
                scale = float(np.std(numeric)) if len(numeric) else 1.0
                self.numeric_stats[attr] = (center, max(scale, 1e-6))
                continue
            counts = values.value_counts(dropna=True)
            categories = list(counts.index)
            self.category_maps[attr] = {value: i + 1 for i, value in enumerate(categories)}
            total = max(int(counts.sum()), 1)
            self.category_freq[attr] = {
                value: float(count / total) for value, count in counts.items()
            }

    def _encode_scalar_value(self, attr, value):
        if pd.isna(value):
            return 0.0, 0.0
        if self.dtypes[attr] == "num":
            center, scale = self.numeric_stats[attr]
            return float(np.clip((float(value) - center) / scale, -8.0, 8.0)), 1.0
        mapping = self.category_maps[attr]
        code = mapping.get(value, 0)
        normalized = float(code / max(len(mapping), 1))
        frequency = self.category_freq[attr].get(value, 0.0)
        return normalized, float(np.log1p(1000.0 * frequency) / np.log(1001.0))

    def _tuple_features(self, observed, embeddings):
        n = len(observed)
        scalar = np.zeros((n, 3 * len(self.attrs)), dtype=np.float32)
        for j, attr in enumerate(self.attrs):
            series = observed[attr]
            missing = series.isna().to_numpy()
            scalar[:, 3 * j + 2] = missing.astype(np.float32)
            for row in np.flatnonzero(~missing):
                value, frequency = self._encode_scalar_value(attr, series.iat[row])
                scalar[row, 3 * j] = value
                scalar[row, 3 * j + 1] = frequency
        projected = np.asarray(self.projector.transform(embeddings), dtype=np.float32)
        return np.concatenate([scalar, projected], axis=1), projected

    def _encode_predicate_value(self, attr, value):
        if isinstance(value, (set, frozenset, list, tuple, np.ndarray)):
            encoded = [self._encode_scalar_value(attr, item)[0] for item in value]
            return float(np.mean(encoded)) if encoded else 0.0
        return self._encode_scalar_value(attr, value)[0]

    def _predicate_features(self, query):
        # Per attribute: present, op, endpoint a, endpoint b.
        features = np.zeros(4 * len(self.attrs), dtype=np.float32)
        position = {attr: i for i, attr in enumerate(self.attrs)}
        for atom in query["phi"].atoms:
            j = position[atom.attr]
            features[4 * j] = 1.0
            features[4 * j + 1] = _OP_CODE[atom.op]
            features[4 * j + 2] = self._encode_predicate_value(atom.attr, atom.a)
            if atom.b is not None:
                features[4 * j + 3] = self._encode_predicate_value(atom.attr, atom.b)
        return features

    def _pair_features(self, tuple_features, tuple_projected, rows, query):
        q_projected = np.asarray(
            self.projector.transform(np.asarray(query["q_e"], dtype=np.float32)[None]),
            dtype=np.float32,
        )[0]
        predicate = self._predicate_features(query)
        count = len(rows)
        return np.concatenate([
            tuple_features[rows],
            np.repeat(predicate[None], count, axis=0),
            np.repeat(q_projected[None], count, axis=0),
            np.abs(tuple_projected[rows] - q_projected),
        ], axis=1).astype(np.float32, copy=False)

    def fit(self, complete_table, observed_table, embeddings, queries):
        """Fit using hidden cells as labels but never as model inputs."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        self._fit_scalar_encoding(observed_table)
        tuple_features, tuple_projected = self._tuple_features(observed_table, embeddings)
        rng = np.random.default_rng(self.seed + 7301)
        per_query = max(1, self.max_train_pairs // max(len(queries), 1))
        feature_blocks = []
        label_blocks = []
        unresolved_total = 0
        for query in queries:
            visible = query["phi"].eval_table(observed_table)
            rows = np.flatnonzero(visible == UNKNOWN)
            unresolved_total += len(rows)
            if len(rows) > per_query:
                rows = np.sort(rng.choice(rows, size=per_query, replace=False))
            if not len(rows):
                continue
            labels = (query["phi"].eval_table(complete_table)[rows] == TRUE).astype(np.int8)
            feature_blocks.append(self._pair_features(
                tuple_features, tuple_projected, rows, query))
            label_blocks.append(labels)
        if not feature_blocks:
            raise ValueError("QueryCls has no unresolved pseudo-missing training pairs")
        X = np.concatenate(feature_blocks, axis=0)
        y = np.concatenate(label_blocks, axis=0)
        if len(X) > self.max_train_pairs:
            rows = np.sort(rng.choice(len(X), size=self.max_train_pairs, replace=False))
            X, y = X[rows], y[rows]
        self.constant_probability = float(y.mean())
        if len(np.unique(y)) > 1:
            self.model = HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=self.max_iter,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=self.seed,
            ).fit(X, y)
        self.manifest = {
            "training_pairs": int(len(y)),
            "unresolved_pairs_before_sampling": int(unresolved_total),
            "positive_rate": float(y.mean()),
            "training_queries": int(len(queries)),
            "hidden_values_used_as_features": False,
            "projection_fit": "training_only",
        }
        return self

    def probabilities(self, observed_table, embeddings, queries):
        """Return per-query joint-satisfaction probabilities for every tuple."""
        tuple_features, tuple_projected = self._tuple_features(observed_table, embeddings)
        output = []
        pending = []
        pending_pairs = 0

        def flush():
            nonlocal pending_pairs
            if not pending:
                return
            features = np.concatenate([item[2] for item in pending], axis=0)
            values = self.model.predict_proba(features)[:, 1].astype(np.float32)
            offset = 0
            for output_index, rows, block in pending:
                end = offset + len(block)
                output[output_index][rows] = values[offset:end]
                offset = end
            pending.clear()
            pending_pairs = 0

        for query in queries:
            visible = query["phi"].eval_table(observed_table)
            # NaN marks a visibly false conjunction and must remain ineligible
            # even at theta=0.  Known TRUE rows are unconditional admissions.
            prob = np.full(len(observed_table), np.nan, dtype=np.float32)
            prob[visible == TRUE] = 1.0
            rows = np.flatnonzero(visible == UNKNOWN)
            output.append(prob)
            if len(rows):
                X = self._pair_features(tuple_features, tuple_projected, rows, query)
                if self.model is None:
                    values = np.full(len(rows), self.constant_probability, dtype=np.float32)
                    prob[rows] = values
                else:
                    if pending and pending_pairs + len(rows) > self.max_predict_pairs:
                        flush()
                    pending.append((len(output) - 1, rows, X))
                    pending_pairs += len(rows)
            # Visible FALSE remains NaN; FALSE dominates any missing atom.
        flush()
        return output

    @staticmethod
    def eligible(probabilities, theta):
        return [prob >= theta for prob in probabilities]
