"""Frozen, source-disjoint, stratified workloads for paper protocol v2."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from mifa.predicates import Atom, Predicate


DEFAULT_BUCKETS = ((0.001, 0.01), (0.01, 0.10), (0.10, 0.50))


def _apportion(n, weights):
    weights = np.asarray(weights, dtype=float)
    raw = n * weights / weights.sum()
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[: n - int(counts.sum())]:
        counts[i] += 1
    return counts


def _schedule(n_queries, rng, n_cat, n_num):
    arities = np.concatenate([
        np.full(count, arity, dtype=int)
        for arity, count in zip((1, 2, 3), _apportion(n_queries, (0.4, 0.4, 0.2)))
    ])
    rng.shuffle(arities)
    type_counts = dict(zip(("categorical", "numerical", "mixed"),
                           _apportion(n_queries, (0.3, 0.3, 0.4))))
    types = np.full(n_queries, "", dtype=object)
    mixed_compatible = np.flatnonzero((arities >= 2) & (arities <= n_cat + n_num))
    if len(mixed_compatible) < type_counts["mixed"] or not n_cat or not n_num:
        raise ValueError("arity schedule cannot accommodate the requested mixed workload")
    # Consume the hardest (largest-arity) mixed slots first. This leaves enough
    # low-arity positions for datasets with only two numerical attributes.
    jitter = rng.random(n_queries)
    mixed_order = mixed_compatible[np.lexsort((jitter[mixed_compatible],
                                                -arities[mixed_compatible]))]
    mixed_slots = mixed_order[:type_counts["mixed"]]
    types[mixed_slots] = "mixed"
    remaining = np.flatnonzero(types == "")
    cat_valid = arities[remaining] <= n_cat
    num_valid = arities[remaining] <= n_num
    mandatory_cat = remaining[~num_valid]
    optional_cat = remaining[cat_valid & num_valid]
    need_optional = type_counts["categorical"] - len(mandatory_cat)
    if (need_optional < 0 or len(optional_cat) < need_optional or
            np.any(arities[mandatory_cat] > n_cat)):
        raise ValueError("dataset attribute inventory cannot realize the declared type/arity marginals")
    chosen_optional = (rng.choice(optional_cat, size=need_optional, replace=False)
                       if need_optional else np.array([], dtype=int))
    categorical_slots = np.concatenate([mandatory_cat, chosen_optional])
    types[categorical_slots] = "categorical"
    numeric_slots = remaining[types[remaining] == ""]
    if (len(numeric_slots) != type_counts["numerical"] or
            np.any(arities[numeric_slots] > n_num)):
        raise ValueError("dataset attribute inventory cannot realize the declared type/arity marginals")
    types[numeric_slots] = "numerical"
    bucket_targets = _apportion(n_queries, (1, 1, 1))
    bucket_ids = np.full(n_queries, -1, dtype=int)
    # Avoid structurally implausible Cartesian cells while preserving all three
    # declared marginals.  High-selectivity conjunctions are often impossible
    # when a categorical atom's largest mass is below sqrt(0.1), while the
    # schedule always has enough single-attribute queries to carry the high
    # bucket.  Three-attribute and then other conjunctive queries carry the low
    # bucket; remaining slots carry the medium bucket.
    preferences = [
        (np.flatnonzero(arities == 1), 2),
        (np.flatnonzero(arities == 3), 0),
        (np.flatnonzero(arities == 2), 0),
    ]
    used = np.zeros(3, dtype=int)
    for slots, preferred_bucket in preferences:
        slots = slots[bucket_ids[slots] < 0]
        rng.shuffle(slots)
        take = min(len(slots), int(bucket_targets[preferred_bucket] - used[preferred_bucket]))
        bucket_ids[slots[:take]] = preferred_bucket
        used[preferred_bucket] += take
    remaining_targets = bucket_targets - np.bincount(
        bucket_ids[bucket_ids >= 0], minlength=3)
    remaining_slots = np.flatnonzero(bucket_ids < 0)
    rest = np.concatenate([
        np.full(count, bucket, dtype=int)
        for bucket, count in enumerate(remaining_targets)
    ])
    rng.shuffle(rest)
    bucket_ids[remaining_slots] = rest
    return list(zip(arities.tolist(), types.tolist(), bucket_ids.tolist()))


def _signature_schedule(n_queries, rng, signature_mix, dtypes,
                        bucket_weights=(0.5, 0.5, 0.0)):
    """Build a frozen application-signature schedule.

    The broad protocol intentionally crosses arity and predicate-type
    marginals.  The canonical-composite protocol instead declares realistic
    application filter signatures (for example category + popularity) before
    any validation/test query is generated.  Selectivity remains stratified,
    with low/medium conjunction buckets balanced by default.
    """
    entries = []
    weights = []
    for item in signature_mix:
        if isinstance(item, dict):
            chosen = tuple(str(attr) for attr in item["attrs"])
            weight = float(item.get("weight", 1.0))
        else:
            chosen = tuple(str(attr) for attr in item)
            weight = 1.0
        if not chosen or len(set(chosen)) != len(chosen):
            raise ValueError("canonical signatures need unique attributes")
        unknown = set(chosen) - set(dtypes)
        if unknown:
            raise ValueError(f"canonical signature has unknown attrs: {sorted(unknown)}")
        if weight <= 0:
            raise ValueError("canonical signature weights must be positive")
        kinds = {dtypes[attr] for attr in chosen}
        predicate_type = ("categorical" if kinds == {"cat"} else
                          "numerical" if kinds == {"num"} else "mixed")
        entries.append((chosen, predicate_type))
        weights.append(weight)
    if not entries:
        raise ValueError("canonical workload needs at least one signature")
    bucket_weights = np.asarray(bucket_weights, dtype=float)
    if bucket_weights.shape != (3,) or np.any(bucket_weights < 0) or \
            bucket_weights.sum() <= 0:
        raise ValueError("canonical bucket weights need three non-negative values")
    schedule = []
    for (chosen, predicate_type), count in zip(
            entries, _apportion(n_queries, weights)):
        bucket_counts = _apportion(int(count), bucket_weights)
        for bucket, bucket_count in enumerate(bucket_counts):
            schedule.extend((chosen, predicate_type, bucket)
                            for _ in range(int(bucket_count)))
    rng.shuffle(schedule)
    return schedule


def _choose_attrs(attrs, dtypes, arity, predicate_type, rng):
    cat = [a for a in attrs if dtypes[a] == "cat"]
    num = [a for a in attrs if dtypes[a] == "num"]
    if predicate_type == "categorical":
        if len(cat) < arity:
            return None
        return rng.choice(cat, size=arity, replace=False).tolist()
    if predicate_type == "numerical":
        if len(num) < arity:
            return None
        return rng.choice(num, size=arity, replace=False).tolist()
    if not cat or not num or arity < 2:
        return None
    chosen = [rng.choice(cat), rng.choice(num)]
    pool = [a for a in attrs if a not in chosen]
    if len(pool) < arity - 2:
        return None
    if arity > 2:
        chosen.extend(rng.choice(pool, size=arity - 2, replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def _atom_with_mass(table, attr, dtype, mass, rng):
    col = table[attr].dropna()
    if len(col) == 0:
        return None
    if dtype == "cat":
        freq = col.value_counts(normalize=True)
        # Draw among the closest values instead of always selecting one constant.
        near = np.argsort(np.abs(freq.to_numpy(dtype=float) - mass))[: min(12, len(freq))]
        value = freq.index[int(rng.choice(near))]
        if isinstance(value, np.generic):
            value = value.item()
        return Atom(attr, "eq", value)
    values = col.to_numpy(dtype=float)
    mass = float(np.clip(mass, 1.0 / max(len(values), 2), 0.95))
    lower_q = float(rng.uniform(0.0, max(1e-9, 1.0 - mass)))
    lo, hi = np.quantile(values, [lower_q, min(1.0, lower_q + mass)])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        return Atom(attr, "le", float(hi))
    return Atom(attr, "between", float(lo), float(hi))


def _predicate_for_stratum(search_spec, attrs, arity, predicate_type, bucket, rng,
                            max_attempts=500):
    low, high = bucket
    for _ in range(max_attempts):
        chosen = _choose_attrs(attrs, search_spec.dtypes, arity, predicate_type, rng)
        if chosen is None:
            break
        target = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        per_atom = target ** (1.0 / arity)
        atoms = [_atom_with_mass(search_spec.table, attr, search_spec.dtypes[attr],
                                 per_atom, rng) for attr in chosen]
        if any(atom is None for atom in atoms):
            continue
        predicate = Predicate(atoms)
        selectivity = predicate.selectivity(search_spec.table)
        if low <= selectivity < high:
            return predicate, selectivity
    raise RuntimeError(
        f"cannot realize workload stratum type={predicate_type} arity={arity} "
        f"selectivity=[{low:.4f},{high:.4f}); audit the dataset attribute inventory"
    )


def _predicate_for_signature(search_spec, chosen, bucket, rng, max_attempts=500):
    low, high = bucket
    chosen = tuple(chosen)
    for _ in range(max_attempts):
        target = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        per_atom = target ** (1.0 / len(chosen))
        atoms = [_atom_with_mass(
            search_spec.table, attr, search_spec.dtypes[attr], per_atom, rng)
            for attr in chosen]
        if any(atom is None for atom in atoms):
            continue
        predicate = Predicate(atoms)
        selectivity = predicate.selectivity(search_spec.table)
        if low <= selectivity < high:
            return predicate, selectivity
    signature = "&".join(sorted(chosen))
    raise RuntimeError(
        f"cannot realize canonical signature={signature} "
        f"selectivity=[{low:.4f},{high:.4f}); audit the training-only inventory"
    )


def sample_stratified_workload(query_spec, search_spec, n_queries, k, *, seed,
                               predicate_attrs=None, buckets=DEFAULT_BUCKETS,
                               predicate_reference_spec=None,
                               signature_mix=None,
                               canonical_bucket_weights=(0.5, 0.5, 0.0)):
    """Generate the declared arity/type/selectivity mix from disjoint query rows.

    Predicate constants and selectivity strata are designed against
    ``predicate_reference_spec``.  Formal held-out runs pass the masked training
    relation here, so workload construction never reads calibration/test hidden
    scalar values.  ``search_spec`` is used only by the evaluator to record the
    realized selectivity of an already frozen predicate.
    """
    allow_source_reuse = bool(query_spec.meta.get("allow_source_reuse", False))
    if n_queries > query_spec.n and not allow_source_reuse:
        raise ValueError("n_queries exceeds the number of disjoint query-source rows")
    if query_spec.n == 0:
        raise ValueError("query source role is empty")
    rng = np.random.default_rng(seed + 99173)
    attrs = list(predicate_attrs or search_spec.query_relevant_attrs)
    if n_queries <= query_spec.n:
        source_rows = rng.choice(query_spec.n, size=n_queries, replace=False)
    else:
        # Multiple predicates per source group are permitted by the paper, but
        # all derivatives of one source stay in exactly one role.  Use balanced
        # shuffled cycles so query-macro and group-macro weighting stay close.
        cycles = []
        remaining = n_queries
        while remaining:
            cycle = rng.permutation(query_spec.n)
            take = min(remaining, query_spec.n)
            cycles.append(cycle[:take])
            remaining -= take
        source_rows = np.concatenate(cycles)
        rng.shuffle(source_rows)
    source_ids = np.asarray(
        query_spec.meta.get("source_ids", np.arange(query_spec.n)), dtype=np.int64)
    if len(source_ids) != query_spec.n:
        raise ValueError("query source_ids must be row-aligned with query embeddings")
    reference = predicate_reference_spec or search_spec
    n_cat = sum(search_spec.dtypes[attr] == "cat" for attr in attrs)
    n_num = sum(search_spec.dtypes[attr] == "num" for attr in attrs)
    schedule = (_signature_schedule(
        n_queries, rng, signature_mix, search_spec.dtypes,
        canonical_bucket_weights) if signature_mix else
        _schedule(n_queries, rng, n_cat, n_num))
    Xq = query_spec.normalized_embeddings()
    queries = []
    predicate_cache = {}
    for src, scheduled in zip(source_rows, schedule):
        if signature_mix:
            chosen, predicate_type, bucket_id = scheduled
            arity = len(chosen)
            key = (tuple(chosen), predicate_type, bucket_id)
        else:
            arity, predicate_type, bucket_id = scheduled
            chosen = None
            key = (arity, predicate_type, bucket_id)
        pool = predicate_cache.setdefault(key, [])
        # Keep a small diverse pool per realizable marginal stratum.  Once it is
        # populated, use a short best-effort refresh and fall back immediately;
        # dependent categorical data can otherwise spend thousands of attempts
        # rediscovering that only a few conjunctions inhabit a narrow bucket.
        should_generate = not pool or (len(pool) < 8 and rng.random() < 0.35)
        try:
            if not should_generate:
                raise RuntimeError("reuse frozen predicate pool")
            if chosen is None:
                predicate, design_selectivity = _predicate_for_stratum(
                    reference, attrs, arity, predicate_type, buckets[bucket_id], rng,
                    max_attempts=1000 if not pool else 200)
            else:
                predicate, design_selectivity = _predicate_for_signature(
                    reference, chosen, buckets[bucket_id], rng,
                    max_attempts=1000 if not pool else 200)
            pool.append((predicate, design_selectivity))
        except RuntimeError:
            # Some dependent categorical inventories have only a small number
            # of realizable conjunctions in a requested marginal stratum.  A
            # frozen predicate may legitimately be paired with multiple query
            # embeddings; reuse an already verified predicate instead of making
            # run success depend on a late random draw.
            if not pool:
                raise
            predicate, design_selectivity = pool[int(rng.integers(len(pool)))]
        realized_selectivity = predicate.selectivity(search_spec.table)
        queries.append({
            "q_e": Xq[src].copy(),
            "k": int(k),
            "phi": predicate,
            "src": int(source_ids[src]),
            "src_local": int(src),
            "source_corpus": query_spec.meta.get("source_corpus", query_spec.name),
            "arity": int(arity),
            "predicate_type": predicate_type,
            "selectivity_bucket": int(bucket_id),
            "selectivity": float(design_selectivity),
            "realized_selectivity": float(realized_selectivity),
            "predicate_reference": "training" if predicate_reference_spec is not None else "search",
            "workload_family": ("canonical_signature" if signature_mix
                                else "stratified"),
        })
    return queries


def _matched_arity_entries(signatures, dtypes):
    """Validate one predeclared signature for each of arities 1, 2, and 3."""
    entries = {}
    for item in signatures:
        chosen = tuple(str(attr) for attr in (
            item["attrs"] if isinstance(item, dict) else item))
        if not chosen or len(set(chosen)) != len(chosen):
            raise ValueError("matched-arity signatures need unique attributes")
        unknown = set(chosen) - set(dtypes)
        if unknown:
            raise ValueError(
                f"matched-arity signature has unknown attrs: {sorted(unknown)}")
        arity = len(chosen)
        if arity not in (1, 2, 3) or arity in entries:
            raise ValueError(
                "matched-arity workload needs exactly one signature at each "
                "of arities 1, 2, and 3")
        kinds = {dtypes[attr] for attr in chosen}
        predicate_type = ("categorical" if kinds == {"cat"} else
                          "numerical" if kinds == {"num"} else "mixed")
        entries[arity] = (chosen, predicate_type)
    if set(entries) != {1, 2, 3}:
        raise ValueError(
            "matched-arity workload needs exactly one signature at each of "
            "arities 1, 2, and 3")
    return [entries[arity] for arity in (1, 2, 3)]


def _matched_predicate_pool(
        reference, complete_relation, chosen, bucket, rng, pool_size):
    """Instantiate on training and measure complete-data selectivity without F1."""
    pool = []
    for _ in range(int(pool_size)):
        predicate, selectivity = _predicate_for_signature(
            reference, chosen, bucket, rng, max_attempts=1000)
        realized = float(predicate.selectivity(complete_relation.table))
        pool.append((predicate, float(selectivity), realized))
    return pool


def _balanced_source_rows(query_spec, count, rng):
    if query_spec.n == 0:
        raise ValueError("query source role is empty")
    allow_reuse = bool(query_spec.meta.get("allow_source_reuse", False))
    if count > query_spec.n and not allow_reuse:
        raise ValueError(
            "matched-arity query count exceeds the disjoint query-source role")
    if count <= query_spec.n:
        return rng.choice(query_spec.n, size=count, replace=False)
    cycles = []
    remaining = int(count)
    while remaining:
        cycle = rng.permutation(query_spec.n)
        take = min(remaining, query_spec.n)
        cycles.append(cycle[:take])
        remaining -= take
    rows = np.concatenate(cycles)
    rng.shuffle(rows)
    return rows


def selectivity_matched_arity_summary(queries):
    """Return the prespecified arity-matching receipt for a frozen workload."""
    matched = [
        query for query in queries
        if query.get("workload_family") == "selectivity_matched_arity"]
    if not matched:
        return None
    by_arity = {}
    realized_by_arity = {}
    for arity in (1, 2, 3):
        selected = [query for query in matched if int(query["arity"]) == arity]
        values = np.asarray(
            [query["selectivity"] for query in selected], dtype=float)
        realized = np.asarray(
            [query["realized_selectivity"] for query in selected], dtype=float)
        if not len(values):
            raise ValueError(f"matched-arity workload has no arity-{arity} rows")
        by_arity[str(arity)] = {
            "query_count": int(len(values)),
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
        }
        realized_by_arity[str(arity)] = {
            "query_count": int(len(realized)),
            "median": float(np.median(realized)),
            "q1": float(np.quantile(realized, 0.25)),
            "q3": float(np.quantile(realized, 0.75)),
        }
    counts = [row["query_count"] for row in by_arity.values()]
    training_medians = np.asarray(
        [row["median"] for row in by_arity.values()])
    realized_medians = np.asarray(
        [row["median"] for row in realized_by_arity.values()])
    training_relative_span = float(
        training_medians.max() / max(training_medians.min(), 1e-12) - 1.0)
    relative_span = float(
        realized_medians.max() / max(realized_medians.min(), 1e-12) - 1.0)
    iqr_overlap = bool(
        max(row["q1"] for row in realized_by_arity.values()) <=
        min(row["q3"] for row in realized_by_arity.values()))
    groups = {}
    for query in matched:
        groups.setdefault(int(query["match_group"]), []).append(query)
    paired = all(
        {int(query["arity"]) for query in group} == {1, 2, 3} and
        len({int(query["src"]) for query in group}) == 1
        for group in groups.values())
    return {
        "matching_source": (
            "complete_data_selectivity_of_training_instantiated_predicates"),
        "selection_uses_test_f1": False,
        "query_counts_equal": len(set(counts)) == 1,
        "paired_query_embedding_rows": bool(paired),
        "training_design_selectivity": by_arity,
        "evaluation_complete_selectivity": realized_by_arity,
        "training_median_relative_span": training_relative_span,
        "median_relative_span": relative_span,
        "median_within_20_percent": bool(relative_span <= 0.20),
        "iqr_overlap": iqr_overlap,
        "acceptance_pass": bool(
            len(set(counts)) == 1 and paired and
            relative_span <= 0.20 and iqr_overlap),
    }


def sample_selectivity_matched_arity_workload(
        query_spec, search_spec, n_queries, k, *, seed, signatures,
        buckets=DEFAULT_BUCKETS, predicate_reference_spec=None,
        bucket_weights=(0.5, 0.5, 0.0), pool_size=96):
    """Build equal, source-paired arity cohorts matched on complete selectivity.

    Predicate constants are instantiated from ``predicate_reference_spec``;
    their complete-data selectivity on the evaluation relation is then used
    for nearest-log-selectivity matching.  No retrieval answer or F1 score
    participates in cohort construction.
    """
    entries = _matched_arity_entries(signatures, search_spec.dtypes)
    if n_queries % len(entries):
        raise ValueError(
            "matched-arity workload query count must be divisible by three")
    per_arity = n_queries // len(entries)
    rng = np.random.default_rng(seed + 99173)
    source_rows = _balanced_source_rows(query_spec, per_arity, rng)
    source_ids = np.asarray(
        query_spec.meta.get("source_ids", np.arange(query_spec.n)), dtype=np.int64)
    if len(source_ids) != query_spec.n:
        raise ValueError("query source_ids must be row-aligned with embeddings")
    weights = np.asarray(bucket_weights, dtype=float)
    if weights.shape != (3,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError(
            "matched-arity bucket weights need three non-negative values")
    bucket_counts = _apportion(per_arity, weights)
    reference = predicate_reference_spec or search_spec
    groups = []
    for bucket_id, count in enumerate(bucket_counts):
        count = int(count)
        if not count:
            continue
        pools = [
            _matched_predicate_pool(
                reference, search_spec, chosen, buckets[bucket_id], rng,
                max(int(pool_size), min(256, count * 2)))
            for chosen, _predicate_type in entries
        ]
        lower = max(
            min(realized for _predicate, _design, realized in pool)
            for pool in pools)
        upper = min(
            max(realized for _predicate, _design, realized in pool)
            for pool in pools)
        reference_values = sorted(
            realized for _predicate, _design, realized in pools[0]
            if lower <= realized <= upper)
        if not reference_values:
            raise RuntimeError(
                "Books cannot realize overlapping complete-data selectivity support "
                f"for matched arity in bucket {bucket_id}")
        target_indices = np.rint(np.linspace(
            0, len(reference_values) - 1, count)).astype(int)
        targets = [reference_values[index] for index in target_indices]
        rng.shuffle(targets)
        for target in targets:
            selected = [
                min(pool, key=lambda item: abs(
                    np.log(max(item[2], 1e-12)) -
                    np.log(max(target, 1e-12))))
                for pool in pools
            ]
            groups.append((bucket_id, float(target), selected))
    if len(groups) != per_arity:
        raise RuntimeError("matched-arity bucket apportionment drifted")
    rng.shuffle(groups)
    Xq = query_spec.normalized_embeddings()
    queries = []
    for match_group, (src, group) in enumerate(zip(source_rows, groups)):
        bucket_id, target, selected = group
        for arity, ((chosen, predicate_type),
                    (predicate, design_selectivity,
                     realized_selectivity)) in enumerate(
                        zip(entries, selected), start=1):
            queries.append({
                "q_e": Xq[src].copy(),
                "k": int(k),
                "phi": predicate,
                "src": int(source_ids[src]),
                "src_local": int(src),
                "source_corpus": query_spec.meta.get(
                    "source_corpus", query_spec.name),
                "arity": int(arity),
                "predicate_type": predicate_type,
                "selectivity_bucket": int(bucket_id),
                "selectivity": float(design_selectivity),
                "realized_selectivity": float(realized_selectivity),
                "predicate_reference": (
                    "training" if predicate_reference_spec is not None
                    else "search"),
                "workload_family": "selectivity_matched_arity",
                "match_group": int(match_group),
                "match_target_selectivity": float(target),
                "signature": "&".join(sorted(chosen)),
            })
    rng.shuffle(queries)
    summary = selectivity_matched_arity_summary(queries)
    if not summary["acceptance_pass"]:
        raise RuntimeError(
            "matched-arity complete-data-selectivity acceptance failed: "
            f"{summary}")
    return queries


def native_filter_workload(query_spec, search_spec, filters, k, *,
                           predicate_reference_spec=None):
    """Build faithful scalar ArXiv native R/EM filters without resampling.

    The benchmark's R family is a range over ``update_date``.  Its EM integer
    label is exactly ``number_of_sub_categories`` (verified against all 10,000
    shipped ground-truth rows).  EMIS is intentionally not scalarized here: it
    is membership in the original multi-valued ``main_categories`` field and
    requires the separate multi-valued-attribute track.
    """
    if len(filters) != query_spec.n:
        raise ValueError("native filters must align with the query embedding rows")
    source_ids = np.asarray(
        query_spec.meta.get("source_ids", np.arange(query_spec.n)), dtype=np.int64)
    Xq = query_spec.normalized_embeddings()
    reference = predicate_reference_spec or search_spec
    queries = []
    for local, item in enumerate(filters):
        family = str(item.get("family", "r")).lower()
        if family == "r":
            predicate = Predicate([
                Atom("update_date", "between", item["range_start"], item["range_end"])
            ])
            predicate_type = "numerical"
        elif family == "em":
            predicate = Predicate([
                Atom("number_of_sub_categories", "eq", int(item["label"]))
            ])
            predicate_type = "numerical"
        else:
            raise ValueError(
                f"unsupported scalar ArXiv native filter family: {family}; "
                "EMIS needs the multi-valued membership track")
        design_selectivity = float(predicate.selectivity(reference.table))
        realized_selectivity = float(predicate.selectivity(search_spec.table))
        if design_selectivity < DEFAULT_BUCKETS[0][1]:
            bucket_id = 0
        elif design_selectivity < DEFAULT_BUCKETS[1][1]:
            bucket_id = 1
        else:
            bucket_id = 2
        queries.append({
            "q_e": Xq[local].copy(),
            "k": int(k),
            "phi": predicate,
            "src": int(source_ids[local]),
            "src_local": int(local),
            "source_corpus": query_spec.meta.get("source_corpus", query_spec.name),
            "arity": 1,
            "predicate_type": predicate_type,
            "selectivity_bucket": int(bucket_id),
            "selectivity": design_selectivity,
            "realized_selectivity": realized_selectivity,
            "predicate_reference": (
                "training" if predicate_reference_spec is not None else "search"),
            "workload_family": f"arxiv_native_{family}",
        })
    return queries


def explicit_native_filter_workload(query_spec, search_spec, filters, k, *,
                                    predicate_reference_spec=None):
    """Build predicates shipped alongside an external query-vector workload.

    Each filter contains an ``atoms`` list whose entries use the repository's
    predicate representation: ``attr``, ``op``, ``a`` and optional ``b``.
    Query vectors and filters remain row-aligned and are never resampled.
    """
    if len(filters) != query_spec.n:
        raise ValueError("explicit filters must align with query embeddings")
    source_ids = np.asarray(
        query_spec.meta.get("source_ids", np.arange(query_spec.n)),
        dtype=np.int64)
    Xq = query_spec.normalized_embeddings()
    reference = predicate_reference_spec or search_spec
    queries = []
    for local, item in enumerate(filters):
        atom_specs = item.get("atoms", ())
        if not atom_specs:
            raise ValueError("an explicit native filter needs at least one atom")
        atoms = []
        for atom_spec in atom_specs:
            attr = str(atom_spec["attr"])
            if attr not in search_spec.dtypes:
                raise ValueError(f"native filter contains unknown attr: {attr}")
            atoms.append(Atom(
                attr, str(atom_spec["op"]), atom_spec.get("a"),
                atom_spec.get("b")))
        predicate = Predicate(atoms)
        kinds = {search_spec.dtypes[atom.attr] for atom in atoms}
        predicate_type = (
            "categorical" if kinds == {"cat"} else
            "numerical" if kinds == {"num"} else "mixed")
        design_selectivity = float(predicate.selectivity(reference.table))
        realized_selectivity = float(predicate.selectivity(search_spec.table))
        if design_selectivity < DEFAULT_BUCKETS[0][1]:
            bucket_id = 0
        elif design_selectivity < DEFAULT_BUCKETS[1][1]:
            bucket_id = 1
        else:
            bucket_id = 2
        queries.append({
            "q_e": Xq[local].copy(),
            "k": int(k),
            "phi": predicate,
            "src": int(source_ids[local]),
            "src_local": int(local),
            "source_corpus": query_spec.meta.get(
                "source_corpus", query_spec.name),
            "arity": len(atoms),
            "predicate_type": predicate_type,
            "selectivity_bucket": int(bucket_id),
            "selectivity": design_selectivity,
            "realized_selectivity": realized_selectivity,
            "predicate_reference": (
                "training" if predicate_reference_spec is not None
                else "search"),
            "workload_family": str(
                item.get("workload_family", "external_native_filter")),
        })
    return queries


def native_range_workload(query_spec, search_spec, filters, k, *,
                          predicate_reference_spec=None):
    """Backward-compatible wrapper for the R-only native workload."""
    return native_filter_workload(
        query_spec, search_spec, filters, k,
        predicate_reference_spec=predicate_reference_spec)


def workload_digest(queries):
    h = hashlib.sha256()
    for query in queries:
        h.update(np.ascontiguousarray(query["q_e"], dtype=np.float32).tobytes())
        payload = {
            key: query[key] for key in
            ("src", "k", "arity", "predicate_type", "selectivity_bucket", "selectivity",
             "realized_selectivity", "predicate_reference")
        }
        payload["source_corpus"] = query.get("source_corpus")
        payload["workload_family"] = query.get("workload_family", "stratified")
        for key in ("match_group", "match_target_selectivity"):
            if key in query:
                payload[key] = query[key]
        payload["atoms"] = [
            {"attr": atom.attr, "op": atom.op, "a": atom.a, "b": atom.b}
            for atom in query["phi"].atoms
        ]
        h.update(json.dumps(payload, sort_keys=True, default=str,
                            separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def workload_counts(queries):
    out = {"arity": {}, "predicate_type": {}, "selectivity_bucket": {},
           "signature": {}}
    for query in queries:
        for key in ("arity", "predicate_type", "selectivity_bucket"):
            value = str(query[key])
            out[key][value] = out[key].get(value, 0) + 1
        signature = "&".join(sorted(atom.attr for atom in query["phi"].atoms))
        out["signature"][signature] = out["signature"].get(signature, 0) + 1
    return out
