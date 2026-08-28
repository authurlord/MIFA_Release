"""Two-level missing-value inference index I_mv (paper §3.2.2), now COARSE-TO-FINE.

For each discovered rule X->Y, group the OBSERVED-complete rows by their X equality
key (categorical raw values, discretized numeric buckets, and/or cell id) into coarse
support groups, and store per-group target-side summaries of Y:
  - categorical Y: empirical {value: probability}
  - numerical Y:   (min, q10, q50, q90, max, n)  (interval + quantiles)
Sigma_ind functional rules collapse to an exact value per X key.

discover_rules now returns an ORDERED LADDER of rules per target (most-specific X∪{cell}
-> coarse X -> arity-1 scalar). I_mv builds group summaries for EACH rung (granularity
LEVEL); a query-time probe maps a tuple's observed evidence to the FINEST level whose
group key exists and returns that level's target domain. This raises coverage: when a
tuple's fine (store,cell) key is unseen, it falls back to the coarse (store) key instead
of dropping to the marginal prior.

SOUNDNESS: a level may hard-certify (exact pass/fail) ONLY if its rung is Sigma_ind
(100%-conf single-valued); a coarse Sigma_dep rung that fires gives an UNCERTAIN domain
with a probability. The fallback picks the finest rung that FIRES, regardless of kind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mifa.inference.rules import _key_frame


class IMV:
    def __init__(self, spec, observed, rules, cell_of=None, n_bins=8, min_group=5,
                 *, support_observed=None, support_cell_of=None):
        """Build support summaries and bind them to a target observed relation.

        Historical callers omit ``support_observed`` and retain the June
        same-relation behavior. Paper-v2 callers pass masked training support
        separately from the validation/calibration/test ``observed`` relation.
        """
        self.spec = spec
        self.dtypes = spec.dtypes
        self.rules = rules
        self.n_bins = n_bins
        self.min_group = min_group
        self.cell_of = cell_of
        support = observed if support_observed is None else support_observed
        support_cells = cell_of if support_cell_of is None else support_cell_of
        # Per target: an ORDERED ladder of granularity levels, finest -> coarsest. Each level:
        #   {"X": tuple(attrs), "ind": bool, "groups": {Xkey(tuple): summary}}
        # ind=True iff the rung is a Sigma_ind functional rule (sound exact pass/fail).
        self._levels = {}     # Y -> list[level dict]
        # cache numeric-evidence bin edges ONCE (avoid O(n^2) quantile recompute)
        self._bins = {}
        for a, dt in self.dtypes.items():
            if dt == "num":
                col = support[a].to_numpy(dtype=float)
                col = col[~np.isnan(col)]
                if len(col):
                    self._bins[a] = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))[1:-1]
        # cache numpy evidence columns + observed masks (avoid pandas .iloc in hot loop)
        self._np = {a: observed[a].to_numpy() for a in observed.columns}
        self._isna = {a: observed[a].isna().to_numpy() for a in observed.columns}
        self._build(support, support_cells)

    def _summary(self, yv, dtype):
        if dtype == "cat":
            vc = pd.Series(yv).value_counts(normalize=True)
            return {"type": "cat", "dist": dict(vc), "n": len(yv)}
        a = np.asarray(yv, dtype=float)
        return {"type": "num", "min": float(a.min()), "max": float(a.max()),
                "q10": float(np.quantile(a, .1)), "q50": float(np.quantile(a, .5)),
                "q90": float(np.quantile(a, .9)), "n": len(a)}

    def _build(self, observed, support_cell_of):
        for Y, rlist in self.rules.items():
            if not rlist:
                continue
            levels = []
            for rule in rlist:                    # ladder order: finest -> coarsest
                X = rule.X
                kf = _key_frame(observed, self.dtypes, list(X), support_cell_of,
                                self.n_bins, self._bins)
                yobs = ~observed[Y].isna()
                kf = kf[yobs.values]; yv = observed[Y][yobs]
                df = kf.copy(); df["_y"] = yv.values
                df = df.dropna(subset=list(X))
                g = {}
                for key, grp in df.groupby(list(X), observed=True):
                    if len(grp) < self.min_group:
                        continue
                    key = key if isinstance(key, tuple) else (key,)
                    g[key] = self._summary(grp["_y"].to_numpy(), self.dtypes[Y])
                if g:
                    levels.append({"X": X, "ind": rule.kind == "ind", "groups": g})
            if levels:
                self._levels[Y] = levels

    def _level_key_for(self, X, t_id):
        """Equality key of tuple t for evidence X, from observed values; None if any
        evidence attr is itself missing (cannot probe at this granularity)."""
        parts = []
        for a in X:
            if a == "cell":
                parts.append(self.cell_of[t_id]); continue
            if self._isna[a][t_id]:
                return None
            v = self._np[a][t_id]
            if self.dtypes[a] == "num":
                edges = self._bins.get(a, np.array([]))   # cached bin edges
                parts.append(int(np.digitize([float(v)], edges)[0]))
            else:
                parts.append(v)
        return tuple(parts)

    def domain(self, observed, t_id, Y):
        """Return inferred domain for t[Y] using the FINEST level whose group key exists
        for t, else the next coarser, else None:
            ('exact',v) | ('set',{v:p}) | ('interval',(lo,hi),summ) | None.
        A level hard-certifies (exact) only if it is Sigma_ind AND single-valued."""
        levels = self._levels.get(Y)
        if not levels:
            return None
        for lvl in levels:
            key = self._level_key_for(lvl["X"], t_id)
            if key is None:
                continue
            summ = lvl["groups"].get(key)
            if summ is None:
                continue
            if summ["type"] == "cat":
                dist = summ["dist"]
                if lvl["ind"] and len(dist) == 1:
                    return ("exact", next(iter(dist)))
                return ("set", dist)
            # numerical
            if lvl["ind"]:
                return ("exact", summ["q50"])
            return ("interval", (summ["q10"], summ["q90"]), summ)
        return None

    def domain_arrays(self, observed, Y):
        """VECTORIZED per-tuple coarse-to-fine inference for target Y. For EACH tuple,
        resolve the FINEST level whose group key matched (one hash-join per level, then a
        precedence-merge: a finer level overwrites a coarser match). Returns arrays aligned
        to all n tuples consumed by certificate_fast/certificate_batch:
          'has'      (n,) bool  -- some level matched
          'gid'      (n,) int   -- index into the COMBINED group list (-1 where !has)
          'is_exact' (n_groups,) bool -- group hard-certifies (Sigma_ind single-valued)
        categorical:
          'dists'    list[dict] (per combined group)  group value->prob
          'gsize','gtop'  per group; 'gprob','val_index'  batch prob matrix
        numerical:
          'lo','hi','exact' (n,) float  -- per-tuple q10/q90/q50 of the resolved group
        None if no level for Y. NOTE: exactness is now PER-GROUP ('is_exact'), since the
        finest firing level may be Sigma_dep while a coarser one is Sigma_ind (or vice
        versa); certificate_fast/batch read is_exact[gid] rather than a global flag."""
        levels = self._levels.get(Y)
        if not levels:
            return None
        is_cat = self.dtypes[Y] == "cat"
        n = len(observed)

        # Combined group list across levels, plus per-group is_exact. Resolve per-tuple gid
        # by walking levels from COARSEST to FINEST so a finer match overwrites a coarser.
        combined_summ = []            # list of summary dicts (the resolved group payloads)
        combined_exact = []           # parallel is_exact flags
        gid = np.full(n, -1, dtype=np.int64)

        for lvl in reversed(levels):  # coarsest -> finest; finest assignment wins
            X = lvl["X"]; groups = lvl["groups"]; ind = lvl["ind"]
            if not groups:
                continue
            kf = _key_frame(observed, self.dtypes, list(X), self.cell_of,
                            self.n_bins, self._bins).copy()
            kf["__row__"] = np.arange(n)
            Xl = list(X)
            grows = []
            base = len(combined_summ)
            for li, (key, s) in enumerate(groups.items()):
                kt = key if isinstance(key, tuple) else (key,)
                d = {a: kt[j] for j, a in enumerate(Xl)}
                d["__cgid__"] = base + li
                grows.append(d)
                combined_summ.append(s)
                if is_cat:
                    combined_exact.append(ind and len(s["dist"]) == 1)
                else:
                    combined_exact.append(ind)
            gdf = pd.DataFrame(grows)
            merged = kf.merge(gdf, on=Xl, how="left").sort_values("__row__")
            cg = merged["__cgid__"].to_numpy()
            matched = ~np.isnan(cg)
            # this level overwrites for every tuple it matched (finer levels processed later)
            cg_int = np.where(matched, np.nan_to_num(cg, nan=0).astype(np.int64), -1)
            gid = np.where(matched, cg_int, gid)

        has = gid >= 0
        is_exact = np.array(combined_exact, dtype=bool)
        ng = len(combined_summ)
        gidc = np.clip(gid, 0, max(ng - 1, 0))
        # PER-TUPLE exactness ('ind_tuple'): the resolved finest-firing group hard-certifies.
        # Backward-compat scalar 'ind' = ALL matched groups are exact (legacy single-level
        # consumers read this; with one Sigma_ind level it reduces to the old flag). The
        # CUDA/numba kernels read 'ind_tuple' (per-tuple) so multi-level stays bit-exact.
        ind_tuple = has & is_exact[gidc]
        scalar_ind = bool(ind_tuple[has].all()) if has.any() else False
        out = {"has": has, "gid": gid, "is_exact": is_exact,
               "ind_tuple": ind_tuple, "ind": scalar_ind}
        if is_cat:
            cat_dists = [s["dist"] for s in combined_summ]
            out["dists"] = cat_dists
            out["gsize"] = np.array([len(d) for d in cat_dists], dtype=np.int64) if cat_dists \
                else np.zeros(0, dtype=np.int64)
            out["gtop"] = np.array([next(iter(d)) for d in cat_dists], dtype=object) if cat_dists \
                else np.zeros(0, dtype=object)
            # batch prob matrix gprob[g, v] = P(Y=val_list[v] | group g)
            val_index = {}
            for d in cat_dists:
                for v in d:
                    if v not in val_index:
                        val_index[v] = len(val_index)
            nv = len(val_index)
            gprob = np.zeros((ng, nv), dtype=np.float64)
            for gi2, d in enumerate(cat_dists):
                for v, p in d.items():
                    gprob[gi2, val_index[v]] = p
            out["val_index"] = val_index
            out["gprob"] = gprob
        else:
            lo = np.full(n, np.nan); hi = np.full(n, np.nan); ex = np.full(n, np.nan)
            if ng:
                q10 = np.array([s["q10"] for s in combined_summ])
                q90 = np.array([s["q90"] for s in combined_summ])
                q50 = np.array([s["q50"] for s in combined_summ])
                idx = np.where(has)[0]; gg = gidc[idx]
                lo[idx] = q10[gg]; hi[idx] = q90[gg]; ex[idx] = q50[gg]
            out["lo"] = lo; out["hi"] = hi; out["exact"] = ex
        return out
