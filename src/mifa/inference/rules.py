"""Rule discovery for MIFA inference (paper §3.2.1): Sigma_mv = Sigma_ind ∪ Sigma_dep.

A rule has the form X -> Y, where Y is a (possibly missing) scalar target and X is an
evidence set drawn from OTHER observed scalar attributes (categorical or discretized
numeric) and optionally the embedding-cell id. Discovery is a bottom-up, level-wise
lattice search (CFD/DD-style), evaluated on the OBSERVED-complete rows only (rows where
X and Y are all non-missing) -- no leakage of masked values.

  - Sigma_ind: single-attribute, 100%-confidence functional rules (e.g. zip->city).
    Applied to a single tuple with no neighbours.
  - Sigma_dep: the rest with confidence >= dep_conf; applied via the I_mv index.

Confidence:
  - categorical Y: support-weighted average, over groups, of the group's max-class
    probability (i.e. accuracy of predicting Y by the group's majority).
  - numerical Y: variance reduction 1 - E[Var(Y|X)]/Var(Y)  (interval tightness).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class Rule:
    X: tuple              # evidence attrs (may include the literal 'cell')
    Y: str                # target attr
    kind: str             # 'ind' | 'dep'
    confidence: float
    support: int
    n_groups: int = 0
    meta: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Rule({'+'.join(self.X)}->{self.Y}, {self.kind}, conf={self.confidence:.3f}, sup={self.support})"


def _discretize(col, n_bins=8, edges=None):
    """Quantile-bucket a numeric column into integer codes (for equality grouping)."""
    vals = col.to_numpy(dtype=float)
    if edges is None:
        observed = vals[~np.isnan(vals)]
        edges = np.unique(np.quantile(observed, np.linspace(0, 1, n_bins + 1)))[1:-1] \
            if len(observed) else np.array([])
    codes = np.digitize(vals, np.asarray(edges, dtype=float))
    return codes


def _key_frame(observed, dtypes, evidence_attrs, cell_of=None, n_bins=8, bin_edges=None):
    """Build a frame of equality-keys for each candidate evidence attr (+ cell)."""
    keys = {}
    for a in evidence_attrs:
        if a == "cell":
            keys[a] = pd.Series(cell_of, index=observed.index)
        elif dtypes[a] == "num":
            edges = None if bin_edges is None else bin_edges.get(a)
            keys[a] = pd.Series(_discretize(observed[a], n_bins, edges), index=observed.index)
        else:
            keys[a] = observed[a]
    # A single-attribute workload has no scalar evidence columns after the
    # target is excluded.  Preserve row alignment for that empty frame so the
    # observed-target mask can still be applied; the discovery loop will then
    # correctly return no scalar-body rules.
    return pd.DataFrame(keys, index=observed.index)


def _eval_rule(kf, y, dtype, Xset, min_support, min_group):
    """Return (confidence, support, n_groups) for X->Y on observed-complete rows."""
    sub = kf[list(Xset)].copy()
    sub["_y"] = y.values
    sub = sub.dropna()
    if len(sub) < min_support:
        return 0.0, 0, 0
    grp = sub.groupby(list(Xset), observed=True)
    tot = len(sub)
    conf_acc, supp, ngroups = 0.0, 0, 0
    if dtype == "cat":
        for _, g in grp:
            m = len(g)
            if m < min_group:
                continue
            top = g["_y"].value_counts().iloc[0]
            conf_acc += top                  # correct if predict majority
            supp += m
            ngroups += 1
        conf = conf_acc / supp if supp else 0.0
    else:
        var_tot = np.var(sub["_y"].to_numpy(dtype=float)) or 1e-9
        wvar = 0.0
        for _, g in grp:
            m = len(g)
            if m < min_group:
                continue
            wvar += m * np.var(g["_y"].to_numpy(dtype=float))
            supp += m
            ngroups += 1
        conf = max(0.0, 1.0 - (wvar / supp) / var_tot) if supp else 0.0
    return conf, supp, ngroups


def discover_rules(spec, observed, cell_of=None, targets=None, max_X=2,
                   min_support=30, min_group=5, dep_conf=0.30, ind_conf=0.999,
                   n_bins=8, coarse_to_fine=True, coarse_cat_min_conf=0.6):
    """Discover Sigma_ind/Sigma_dep. Evidence = all scalar cols (≠Y) + 'cell'.

    Returns dict: target -> ORDERED ladder list[Rule], from MOST-SPECIFIC (finest key,
    highest confidence, may include 'cell') to MOST-GENERAL (coarsest key, highest
    coverage), so I_mv/the certificate can fall back: fine (X∪{cell}) -> coarse (X) ->
    arity-1 scalar -> marginal prior (paper §3.2 "value abstraction").

    The ladder is intentionally CONSERVATIVE: Sigma_ind functional rules first (sound
    exact pass/fail), then the best (X+cell) Sigma_dep rule (most specific), then the
    best arity-1 SCALAR (no-cell) Sigma_dep rule (highest coverage). Each rung is a
    distinct GRANULARITY LEVEL keyed by its own X; at query time the engine picks the
    finest rung whose group key exists for a tuple. `coarse_to_fine=False` restores the
    legacy single-best-dep behaviour (for ablation / opt-out).

    REGRESSION GUARD (`coarse_cat_min_conf`): a CATEGORICAL coarse scalar rung is added
    ONLY if its confidence clears this floor. A weak coarse rung (e.g. movielens
    top_tag->primary_genre, conf~0.55) fires for fine-key-miss tuples and replaces the
    marginal-prior fallback with a noisier set-distribution -> it cost ~1pt recall at low
    rho. Strong coarse FDs (yelp city->state 0.998, amazon store->main_category 0.937)
    clear the floor and keep their large gains. NUMERIC coarse rungs are NOT gated (their
    'confidence' is variance-reduction on a different scale; none regressed)."""
    targets = targets or spec.query_relevant_attrs
    scalar_attrs = [c for c in spec.table.columns]
    cand_evidence = list(scalar_attrs) + (["cell"] if cell_of is not None else [])
    rules = {}
    for Y in targets:
        ev = [a for a in cand_evidence if a != Y]
        kf = _key_frame(observed, spec.dtypes, ev, cell_of, n_bins)
        y = observed[Y]
        # mask to rows where Y observed
        yobs = ~y.isna()
        kf_y = kf[yobs.values]
        yv = y[yobs]
        ind_rules = []
        best_dep = None                 # best dep overall (legacy / most-specific rung)
        best_dep_cell = None            # best dep whose X INCLUDES 'cell' (finest)
        best_dep_scalar1 = None         # best arity-1 SCALAR dep, no cell (coarsest, high coverage)
        # level-wise: size 1 then 2
        considered = []
        for size in range(1, max_X + 1):
            for Xset in combinations(ev, size):
                # conciseness: skip if a size-1 subset already gave a 100% rule
                if size > 1 and any(set(s).issubset(Xset) for s in considered):
                    continue
                conf, supp, ng = _eval_rule(kf_y, yv, spec.dtypes[Y], Xset,
                                            min_support, min_group)
                if supp < min_support:
                    continue
                if size == 1 and conf >= ind_conf:
                    ind_rules.append(Rule(tuple(Xset), Y, "ind", conf, supp, ng))
                    considered.append(Xset)
                elif conf >= dep_conf:
                    r = Rule(tuple(Xset), Y, "dep", conf, supp, ng)
                    if best_dep is None or conf > best_dep.confidence:
                        best_dep = r
                    has_cell = "cell" in Xset
                    if has_cell:
                        if best_dep_cell is None or conf > best_dep_cell.confidence:
                            best_dep_cell = r
                    elif size == 1:
                        # categorical coarse rung must clear the regression-guard floor; numeric
                        # (variance-reduction conf) is not gated (see docstring).
                        if spec.dtypes[Y] == "cat" and conf < coarse_cat_min_conf:
                            continue
                        if best_dep_scalar1 is None or conf > best_dep_scalar1.confidence:
                            best_dep_scalar1 = r
        # Assemble the ordered ladder (finest -> coarsest). Sigma_ind rungs always first
        # (sound exact). Then dep rungs by specificity, skipping duplicates by X.
        found = list(ind_rules)
        if coarse_to_fine:
            ladder_dep = []
            for r in (best_dep_cell, best_dep, best_dep_scalar1):
                if r is None:
                    continue
                if any(r.X == e.X for e in ladder_dep):
                    continue
                ladder_dep.append(r)
            # finest first: a rung containing 'cell' is more specific than one without;
            # among same cell-status, more evidence attrs (longer X) is finer.
            ladder_dep.sort(key=lambda r: (("cell" in r.X), len(r.X)), reverse=True)
            found.extend(ladder_dep)
        else:
            if best_dep is not None:
                found.append(best_dep)
        rules[Y] = found
    return rules
