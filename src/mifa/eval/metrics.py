"""Effectiveness metrics vs Ans*(q, D*) + work counters.

recall@k / precision@k are computed against the exact reference answer set.
We also report false-negative rate and false-positive rate explicitly, since
they are the paper's motivation.
"""
from __future__ import annotations

import numpy as np


def recall_at_k(returned_ids, truth_ids):
    truth = set(int(i) for i in truth_ids)
    if not truth:
        return 1.0  # vacuous: no ground-truth answers
    ret = set(int(i) for i in returned_ids)
    return len(ret & truth) / len(truth)


def precision_at_k(returned_ids, truth_ids):
    ret = list(int(i) for i in returned_ids)
    if not ret:
        return 1.0
    truth = set(int(i) for i in truth_ids)
    return sum(1 for i in ret if i in truth) / len(ret)


def fn_fp_rates(returned_ids, truth_ids):
    """false-negative rate = frac of truth missed; false-positive rate = frac of
    returned that are not in truth."""
    truth = set(int(i) for i in truth_ids)
    ret = set(int(i) for i in returned_ids)
    fn = (len(truth - ret) / len(truth)) if truth else 0.0
    fp = (len(ret - truth) / len(ret)) if ret else 0.0
    return fn, fp


def aggregate(per_query):
    """per_query: list of dicts with recall/precision/fn/fp/n_returned. Mean dict."""
    keys = ["recall", "precision", "fn_rate", "fp_rate", "n_returned", "fill_rate"]
    out = {}
    for key in keys:
        vals = [q[key] for q in per_query if key in q]
        out[key] = float(np.mean(vals)) if vals else None
    out["n_queries"] = len(per_query)
    return out
