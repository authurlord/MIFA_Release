"""Shared attribute-level support index for paper-aligned MIFA inference."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _count_joint_interval(order, begin, end, support_mask, satisfied):
    denominator = 0
    joint = 0
    for position in range(begin, end):
        row = order[position]
        if support_mask[row]:
            denominator += 1
            if satisfied[row]:
                joint += 1
    return denominator, joint


@njit(cache=True, fastmath=False)
def _leaf_prefix_joint_counts(leaf_ids, support_mask, satisfied, n_leaves):
    """Aggregate one predicate's evidence once for every DFS leaf."""
    denominator = np.zeros(n_leaves + 1, dtype=np.int64)
    joint = np.zeros(n_leaves + 1, dtype=np.int64)
    for row in range(len(leaf_ids)):
        if not support_mask[row]:
            continue
        slot = int(leaf_ids[row]) + 1
        denominator[slot] += 1
        if satisfied[row]:
            joint[slot] += 1
    for slot in range(1, n_leaves + 1):
        denominator[slot] += denominator[slot - 1]
        joint[slot] += joint[slot - 1]
    return denominator, joint


class SharedSupportIndex:
    """Postings shared across templates and hierarchical context nodes."""

    def __init__(self, table, dtypes, leaf_ids, tree):
        self.table = table.reset_index(drop=True)
        self.dtypes = dict(dtypes)
        self.leaf_ids = np.ascontiguousarray(leaf_ids, dtype=np.int64)
        self.tree = tree
        if self.leaf_ids.shape != (len(self.table),):
            raise ValueError("leaf_ids must align with support rows")
        if len(self.leaf_ids) and (
                self.leaf_ids.min() < 0 or self.leaf_ids.max() >= tree.n_leaves):
            raise ValueError("support leaf ID outside the context tree")
        self.availability = {}
        self.availability_mask = {}
        self.categorical = {}
        self.numerical = {}
        self.columns = {}
        for attr in self.table.columns:
            observed = self.table[attr].notna().to_numpy()
            rows = np.flatnonzero(observed).astype(np.int64)
            self.availability[attr] = rows
            self.availability_mask[attr] = np.ascontiguousarray(
                observed, dtype=np.bool_)
            self.columns[attr] = self.table[attr].to_numpy()
            if self.dtypes.get(attr) == "num":
                values = self.table[attr].to_numpy(dtype=float)[rows]
                order = np.lexsort((rows, values))
                self.numerical[attr] = (values[order], rows[order])
            else:
                postings = defaultdict(list)
                values = self.table[attr].to_numpy()
                for row in rows:
                    postings[values[row]].append(int(row))
                self.categorical[attr] = {
                    value: np.asarray(ids, dtype=np.int64)
                    for value, ids in postings.items()
                }
        # A node covers one contiguous interval of leaf IDs.  Sorting support
        # rows once by leaf lets query-time retrieval start from that interval
        # and apply attribute filters with vectorized masks.  It retains the
        # paper's shared attribute representation (no template--node support
        # lists) while avoiding repeated general-purpose set intersections.
        self._leaf_order = np.argsort(self.leaf_ids, kind="stable").astype(
            np.int64, copy=False)
        leaf_counts = np.bincount(
            self.leaf_ids, minlength=self.tree.n_leaves).astype(np.int64)
        self._leaf_offsets = np.empty(self.tree.n_leaves + 1, dtype=np.int64)
        self._leaf_offsets[0] = 0
        np.cumsum(leaf_counts, out=self._leaf_offsets[1:])
        # Compile the interval counter while constructing the index.
        _count_joint_interval(
            self._leaf_order, 0, 0,
            np.zeros(len(self.table), dtype=np.bool_),
            np.zeros(len(self.table), dtype=np.bool_))
        _leaf_prefix_joint_counts(
            self.leaf_ids,
            np.zeros(len(self.table), dtype=np.bool_),
            np.zeros(len(self.table), dtype=np.bool_), self.tree.n_leaves)
        self._target_availability_masks = {}
        # Body filters are independent of the query predicate.  Reuse their
        # immutable dense masks across predicates instead of rebuilding and
        # copying an O(|D|) boolean vector for every statistical cache miss.
        self._body_mask_cache = {}
        self._body_mask_packed_cache = {}
        self._body_mask_denominator_cache = {}
        self._body_mask_cache_max_entries = min(
            16384, max(2048, (128 * 1024 * 1024) // max(len(self.table), 1)))
        self._body_mask_prebuilt_entries = 0
        self.stats = {"requests": 0, "postings": 0, "intersections": 0,
                      "candidate_rows": 0, "body_mask_cache_hits": 0,
                      "body_mask_cache_misses": 0,
                      "body_mask_cache_resets": 0}

    @staticmethod
    def _cache_value(value):
        if isinstance(value, np.generic):
            value = value.item()
        try:
            hash(value)
            return value
        except TypeError:
            return repr(value)

    @staticmethod
    def _missing_cache_value(value):
        return value is None or (
            isinstance(value, (float, np.floating)) and np.isnan(value))

    @staticmethod
    def pack_mask(mask):
        """Pack a dense boolean mask into aligned uint64 words."""
        packed = np.packbits(
            np.asarray(mask, dtype=np.bool_), bitorder="little")
        remainder = len(packed) % 8
        if remainder:
            packed = np.pad(packed, (0, 8 - remainder))
        return np.ascontiguousarray(packed).view(np.uint64)

    def prebuild_body_masks(
            self, templates, target_columns, *, max_signatures=None):
        """Materialize low-cardinality scalar-body masks at index build time.

        These masks depend only on immutable support/target scalar values and
        the deployed template definitions, never on an online predicate or
        hidden value.  High-cardinality bodies stay on the bounded lazy path
        to avoid quadratic index growth.
        """
        if max_signatures is None:
            # Bound one matching level to roughly 64 MiB of dense masks while
            # allowing broader prebuild coverage on smaller support tables.
            max_signatures = min(
                4096, max(512, (64 * 1024 * 1024) // max(len(self.table), 1)))
        before = len(self._body_mask_cache)
        satisfied = np.zeros(len(self.table), dtype=np.bool_)
        for template in templates:
            if not template.body:
                continue
            level_configs = []
            levels = {body.attr: 0 for body in template.body}
            level_configs.append(dict(levels))
            body_by_name = {body.attr: body for body in template.body}
            for attr in template.scalar_relax_order:
                body = body_by_name[attr]
                while levels[attr] + 1 < len(body.levels):
                    levels = dict(levels)
                    levels[attr] += 1
                    level_configs.append(dict(levels))
            for level_by_attr in level_configs:
                representatives = {}
                for tuple_id in range(len(next(iter(target_columns.values())))):
                    values = tuple(
                        target_columns[body.attr][tuple_id]
                        for body in template.body)
                    if any(self._missing_cache_value(value)
                           for value in values):
                        continue
                    signature = tuple(self._cache_value(value)
                                      for value in values)
                    representatives.setdefault(signature, tuple_id)
                    if len(representatives) > int(max_signatures):
                        representatives = None
                        break
                if representatives is None:
                    continue
                for tuple_id in representatives.values():
                    self.count_joint_indexed(
                        template, target_columns, tuple_id, satisfied,
                        node_id=self.tree.root_id,
                        level_by_attr=level_by_attr, mask_cache={},
                        cache_body=True, record_stats=False)
        self._body_mask_prebuilt_entries = max(
            self._body_mask_prebuilt_entries,
            len(self._body_mask_cache) - before)
        return int(len(self._body_mask_cache) - before)

    @staticmethod
    def _intersection(arrays):
        arrays = [np.asarray(array, dtype=np.int64) for array in arrays]
        if not arrays:
            return np.empty(0, dtype=np.int64)
        arrays.sort(key=len)
        # Every resident posting is constructed from distinct row IDs.  The
        # old path nevertheless re-hashed each posting with ``np.unique`` and
        # asked ``intersect1d`` to deduplicate again on every support request.
        # Preserve the exact set intersection while avoiding that redundant
        # allocation and hashing work.
        result = arrays[0]
        for array in arrays[1:]:
            result = np.intersect1d(result, array, assume_unique=True)
            if not len(result):
                break
        return result

    def predicate_satisfaction_mask(self, atoms):
        """Vectorized TRUE mask for a residual conjunction on support rows."""
        satisfied = np.ones(len(self.table), dtype=np.bool_)
        for atom in atoms:
            values = self.columns[atom.attr]
            satisfied &= self.availability_mask[atom.attr]
            # Missing numerical entries are already excluded by availability;
            # silence the expected invalid comparison at their NaN positions.
            with np.errstate(invalid="ignore"):
                if atom.op == "le":
                    satisfied &= values <= atom.a
                elif atom.op == "ge":
                    satisfied &= values >= atom.a
                elif atom.op == "between":
                    satisfied &= values >= atom.a
                    satisfied &= values <= atom.b
                elif atom.op == "eq":
                    satisfied &= values == atom.a
                elif atom.op == "in":
                    satisfied &= np.isin(values, tuple(atom.a))
                else:
                    raise ValueError(
                        f"unsupported predicate operation: {atom.op}")
            if not satisfied.any():
                break
        return np.ascontiguousarray(satisfied, dtype=np.bool_)

    def _body_posting(self, body, level, value):
        match = body.levels[int(level)]
        if match.kind == "all":
            return self.availability[body.attr]
        if match.kind == "eq":
            return self.categorical.get(body.attr, {}).get(
                value, np.empty(0, dtype=np.int64))
        if match.kind == "radius":
            values, rows = self.numerical[body.attr]
            lower = float(value) - float(match.radius)
            upper = float(value) + float(match.radius)
            start = np.searchsorted(values, lower, side="left")
            end = np.searchsorted(values, upper, side="right")
            return np.sort(rows[start:end])
        raise ValueError(f"unsupported scalar match level: {match.kind}")

    def retrieve(self, template, target_row, *, node_id, level_by_attr):
        postings = [self.availability[attr] for attr in template.targets]
        for body in template.body:
            postings.append(self._body_posting(
                body, level_by_attr[body.attr], target_row[body.attr]))
        self.stats["requests"] += 1
        self.stats["postings"] += len(postings)
        self.stats["intersections"] += max(len(postings) - 1, 0)
        if not postings:
            rows = np.arange(len(self.table), dtype=np.int64)
        else:
            rows = self._intersection(postings)
        node = self.tree.nodes[int(node_id)]
        if len(rows):
            leaf = self.leaf_ids[rows]
            rows = rows[(leaf >= node.leaf_lo) & (leaf <= node.leaf_hi)]
        self.stats["candidate_rows"] += int(len(rows))
        return rows

    def retrieve_indexed(self, template, target_columns, tuple_id, *, node_id,
                         level_by_attr):
        """Retrieve support from a leaf interval using vectorized shared filters."""
        node = self.tree.nodes[int(node_id)]
        begin = int(self._leaf_offsets[node.leaf_lo])
        end = int(self._leaf_offsets[node.leaf_hi + 1])
        rows = self._leaf_order[begin:end]
        keep = np.ones(len(rows), dtype=np.bool_)
        filters = 0
        for attr in template.targets:
            keep &= self.availability_mask[attr][rows]
            filters += 1
        for body in template.body:
            level = body.levels[int(level_by_attr[body.attr])]
            keep &= self.availability_mask[body.attr][rows]
            value = target_columns[body.attr][int(tuple_id)]
            if level.kind == "eq":
                keep &= self.columns[body.attr][rows] == value
            elif level.kind == "radius":
                support_values = np.asarray(
                    self.columns[body.attr][rows], dtype=np.float64)
                radius = float(level.radius)
                keep &= support_values >= float(value) - radius
                keep &= support_values <= float(value) + radius
            elif level.kind != "all":
                raise ValueError(f"unsupported scalar match level: {level.kind}")
            filters += 1
        self.stats["requests"] += 1
        self.stats["postings"] += filters
        self.stats["intersections"] += max(filters - 1, 0)
        selected = rows[keep]
        self.stats["candidate_rows"] += int(len(selected))
        return selected

    def count_satisfaction_scan(
            self, template, target_columns, tuple_id, satisfied, *, node_id,
            level_by_attr, record_stats=True):
        """Semantic fallback for ``w/o I_mv``: scan the complete D_tr arrays.

        This deliberately bypasses postings, leaf-order intervals, prefix
        counts, and cached body masks.  It preserves the exact support/context
        definition while exposing the cost of removing the shared support
        index.
        """
        keep = np.ones(len(self.table), dtype=np.bool_)
        filters = 0
        for attr in template.targets:
            keep &= self.availability_mask[attr]
            filters += 1
        for body in template.body:
            level = body.levels[int(level_by_attr[body.attr])]
            keep &= self.availability_mask[body.attr]
            value = target_columns[body.attr][int(tuple_id)]
            if level.kind == "eq":
                keep &= self.columns[body.attr] == value
            elif level.kind == "radius":
                values = np.asarray(self.columns[body.attr], dtype=np.float64)
                radius = float(level.radius)
                keep &= values >= float(value) - radius
                keep &= values <= float(value) + radius
            elif level.kind != "all":
                raise ValueError(f"unsupported scalar match level: {level.kind}")
            filters += 1
        node = self.tree.nodes[int(node_id)]
        keep &= self.leaf_ids >= int(node.leaf_lo)
        keep &= self.leaf_ids <= int(node.leaf_hi)
        denominator = int(np.count_nonzero(keep))
        plus = int(np.count_nonzero(keep & satisfied))
        if record_stats:
            self.stats["requests"] += 1
            self.stats["postings"] += filters
            self.stats["intersections"] += max(filters - 1, 0)
            self.stats["candidate_rows"] += int(len(self.table))
        return denominator, plus

    def count_joint_indexed(
            self, template, target_columns, tuple_id, satisfied, *, node_id,
            level_by_attr, mask_cache, cache_body=True, record_stats=True,
            satisfied_packed=None):
        """Count denominator/joint successes without materializing support IDs.

        The attribute/body mask is identical along an ancestor chain.  A
        caller-local cache builds it once, while each context only slices its
        contiguous leaf interval and performs two reductions.
        """
        level_key = tuple(
            int(level_by_attr[body.attr]) for body in template.body)
        body_key = None
        if template.body:
            body_key = (
                str(template.template_id),
                tuple(template.targets),
                tuple((body.attr, int(level_by_attr[body.attr]),
                       self._cache_value(
                           target_columns[body.attr][int(tuple_id)]))
                      for body in template.body))
        support_mask = mask_cache.get(level_key) if cache_body else None
        filters = len(template.targets) + len(template.body)
        if support_mask is None:
            target_key = tuple(template.targets)
            target_mask = self._target_availability_masks.get(target_key)
            if target_mask is None:
                target_mask = np.ones(len(self.table), dtype=np.bool_)
                for attr in template.targets:
                    target_mask &= self.availability_mask[attr]
                target_mask.setflags(write=False)
                self._target_availability_masks[target_key] = target_mask
            if template.body and cache_body:
                support_mask = self._body_mask_cache.get(body_key)
            if support_mask is None:
                support_mask = (target_mask if not template.body else
                                target_mask.copy())
                for body in template.body:
                    level = body.levels[int(level_by_attr[body.attr])]
                    support_mask &= self.availability_mask[body.attr]
                    value = target_columns[body.attr][int(tuple_id)]
                    if level.kind == "eq":
                        support_mask &= self.columns[body.attr] == value
                    elif level.kind == "radius":
                        support_values = np.asarray(
                            self.columns[body.attr], dtype=np.float64)
                        radius = float(level.radius)
                        support_mask &= support_values >= float(value) - radius
                        support_mask &= support_values <= float(value) + radius
                    elif level.kind != "all":
                        raise ValueError(
                            f"unsupported scalar match level: {level.kind}")
                if body_key is not None:
                    if cache_body:
                        if len(self._body_mask_cache) >= \
                                self._body_mask_cache_max_entries:
                            self._body_mask_cache.clear()
                            self._body_mask_packed_cache.clear()
                            self._body_mask_denominator_cache.clear()
                            if record_stats:
                                self.stats["body_mask_cache_resets"] += 1
                        support_mask.setflags(write=False)
                        self._body_mask_cache[body_key] = support_mask
                        self._body_mask_packed_cache[body_key] = \
                            self.pack_mask(support_mask)
                        self._body_mask_denominator_cache[body_key] = int(
                            np.count_nonzero(support_mask))
                        if record_stats:
                            self.stats["body_mask_cache_misses"] += 1
            elif body_key is not None and record_stats:
                self.stats["body_mask_cache_hits"] += 1
            if cache_body:
                mask_cache[level_key] = support_mask
        node = self.tree.nodes[int(node_id)]
        begin = int(self._leaf_offsets[node.leaf_lo])
        end = int(self._leaf_offsets[node.leaf_hi + 1])
        if (cache_body and body_key is not None and
                int(node_id) == int(self.tree.root_id)
                and body_key in self._body_mask_packed_cache):
            denominator = self._body_mask_denominator_cache[body_key]
            if satisfied_packed is None:
                satisfied_packed = self.pack_mask(satisfied)
            joint = int(np.bitwise_count(
                self._body_mask_packed_cache[body_key] &
                satisfied_packed).sum())
        else:
            denominator, joint = _count_joint_interval(
                self._leaf_order, begin, end, support_mask, satisfied)
        if record_stats:
            self.stats["requests"] += 1
            self.stats["postings"] += filters
            self.stats["intersections"] += max(filters - 1, 0)
            self.stats["candidate_rows"] += denominator
        return denominator, joint

    def leaf_prefix_joint_counts(self, template, satisfied):
        """Build query-local O(1) ancestor counts for a body-free template."""
        if template.body:
            raise ValueError("leaf-prefix counts require a body-free template")
        target_key = tuple(template.targets)
        target_mask = self._target_availability_masks.get(target_key)
        if target_mask is None:
            target_mask = np.ones(len(self.table), dtype=np.bool_)
            for attr in template.targets:
                target_mask &= self.availability_mask[attr]
            target_mask.setflags(write=False)
            self._target_availability_masks[target_key] = target_mask
        return _leaf_prefix_joint_counts(
            self.leaf_ids, target_mask, satisfied, self.tree.n_leaves)

    def count_joint_from_leaf_prefix(
            self, template, node_id, prefix, *, record_stats=True):
        """Return one node's counts from a query-local DFS-leaf prefix."""
        denominator_prefix, joint_prefix = prefix
        node = self.tree.nodes[int(node_id)]
        begin = int(node.leaf_lo)
        end = int(node.leaf_hi) + 1
        denominator = int(
            denominator_prefix[end] - denominator_prefix[begin])
        joint = int(joint_prefix[end] - joint_prefix[begin])
        filters = len(template.targets)
        if record_stats:
            self.stats["requests"] += 1
            self.stats["postings"] += filters
            self.stats["intersections"] += max(filters - 1, 0)
            self.stats["candidate_rows"] += denominator
        return denominator, joint

    @property
    def resident_bytes(self):
        total = self.leaf_ids.nbytes + self._leaf_order.nbytes + \
            self._leaf_offsets.nbytes
        total += sum(array.nbytes for array in self.availability_mask.values())
        total += sum(array.nbytes for array in self.availability.values())
        total += sum(array.nbytes for postings in self.categorical.values()
                     for array in postings.values())
        total += sum(values.nbytes + rows.nbytes
                     for values, rows in self.numerical.values())
        total += sum(mask.nbytes for mask in self._body_mask_cache.values())
        total += sum(mask.nbytes
                     for mask in self._body_mask_packed_cache.values())
        return int(total)

    def manifest(self):
        return {
            "kind": "shared_attribute_postings",
            "rows": int(len(self.table)),
            "categorical_attributes": sorted(self.categorical),
            "numerical_attributes": sorted(self.numerical),
            "target_availability_attributes": sorted(self.availability),
            "body_mask_cache_entries": int(len(self._body_mask_cache)),
            "body_mask_prebuilt_entries": int(
                self._body_mask_prebuilt_entries),
            "body_mask_cache_max_entries": int(
                self._body_mask_cache_max_entries),
            "body_mask_cache_hits": int(
                self.stats["body_mask_cache_hits"]),
            "body_mask_cache_misses": int(
                self.stats["body_mask_cache_misses"]),
            "body_mask_cache_bytes": int(
                sum(mask.nbytes
                    for mask in self._body_mask_cache.values()) +
                sum(mask.nbytes
                    for mask in self._body_mask_packed_cache.values())),
            "resident_bytes": self.resident_bytes,
        }
