"""Rule-based coarse-to-fine certificates (paper §3.2.3).

Given a predicate phi, for each tuple this produces state in {pass, fail, uncertain}
and p_phi(t), by checking the INFERRED DOMAIN of each missing referenced attribute
against the atom's satisfying region:
  - domain ⊆ region            -> atom pass
  - domain ∩ region = ∅        -> atom fail
  - partial overlap            -> atom uncertain, with p = mass(region | domain)
Conjunction: pass iff all atoms pass; fail iff any fails; else uncertain with
p_phi = product of per-atom probabilities.

Inference uses Sigma_ind (exact value) first, then Sigma_dep via I_mv (candidate set
/ interval). This is the RULE-BASED engine -- embedding-kNN is only a baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mifa.predicates import TRUE, FALSE, UNKNOWN


def _atom_region_cat(atom):
    """Return a test fn v->bool for a categorical atom."""
    if atom.op == "eq":
        return lambda v: v == atom.a
    if atom.op == "in":
        S = set(atom.a); return lambda v: v in S
    raise ValueError(f"cat atom op {atom.op}")


def _atom_prob_cat(atom, dist):
    """P(atom satisfied | categorical dist {v:p})."""
    test = _atom_region_cat(atom)
    return float(sum(p for v, p in dist.items() if test(v)))


def _atom_state_num_interval(atom, lo, hi):
    """(state, p_estimate) for numeric atom given inferred interval [lo,hi]."""
    if atom.op == "le":
        if hi <= atom.a: return "pass", 1.0
        if lo > atom.a: return "fail", 0.0
        return "uncertain", (atom.a - lo) / (hi - lo + 1e-9)
    if atom.op == "ge":
        if lo >= atom.a: return "pass", 1.0
        if hi < atom.a: return "fail", 0.0
        return "uncertain", (hi - atom.a) / (hi - lo + 1e-9)
    if atom.op == "between":
        if lo >= atom.a and hi <= atom.b: return "pass", 1.0
        if hi < atom.a or lo > atom.b: return "fail", 0.0
        ov = max(0.0, min(hi, atom.b) - max(lo, atom.a))
        return "uncertain", ov / (hi - lo + 1e-9)
    raise ValueError(atom.op)


class RuleEngine:
    def __init__(self, spec, observed, imv, refiner=None):
        self.spec = spec
        self.obs = observed
        self.imv = imv
        self.refiner = refiner          # optional embedding fallback when no rule fires
        self.dtypes = spec.dtypes
        self._np = {a: observed[a].to_numpy() for a in observed.columns}
        self._isna = {a: observed[a].isna().to_numpy() for a in observed.columns}

    def _atom_for_tuple(self, atom, t_id, prior):
        """Return (state, p) for one atom on one tuple using observed value or rules.
        `prior` = marginal P(atom satisfied) over observed values, used as the
        fallback when no rule evidence is available (paper: empirical I_mv stats)."""
        if not self._isna[atom.attr][t_id]:     # observed -> hard decision
            ok = atom.eval_value(self._np[atom.attr][t_id])
            return ("pass", 1.0) if ok == TRUE else ("fail", 0.0)
        dom = self.imv.domain(self.obs, t_id, atom.attr)
        if dom is None:
            return ("uncertain", prior)         # no rule evidence -> marginal prior
        # Hard pass/fail certificates may be
        # issued ONLY by tuple-INDEPENDENT (Sigma_ind, 100%-conf exact) rules -- those are the
        # only sound ones, as the paper's predicate-safety guarantee requires. Tuple-DEPENDENT
        # (Sigma_dep, statistical) evidence is NOT a hard bound (q10/q90 quantiles, partial
        # support), so it must stay UNCERTAIN with a probability p_phi; admission is then governed
        # by theta. Previously Sigma_dep could hard-fail/pass -> ~9% unsound on real data.
        if dom[0] == "exact":               # only returned for Sigma_ind rules
            ok = atom.eval_value(dom[1])
            return ("pass", 1.0) if ok == TRUE else ("fail", 0.0)
        if dom[0] == "set":                 # Sigma_dep categorical distribution -> uncertain
            return ("uncertain", _atom_prob_cat(atom, dom[1]))
        # Sigma_dep numeric interval -> uncertain; p estimated from the (q10,q90) band, but
        # the band is NOT used to hard-certify fail/pass.
        lo, hi = dom[1]
        _, p = _atom_state_num_interval(atom, lo, hi)
        return ("uncertain", float(p))

    def certificate(self, predicate, theta=0.5):
        """States + p_phi over all tuples. Observed-decidable tuples are vectorized;
        only tuples with a MISSING referenced attr enter the per-tuple inference loop."""
        from mifa.predicates import Predicate
        n = len(self.obs); atoms = predicate.atoms
        obs_sat, obs_mask, priors = {}, {}, {}
        hard_fail = np.zeros(n, dtype=bool)
        any_missing = np.zeros(n, dtype=bool)
        for a in atoms:
            om = ~self._isna[a.attr]; obs_mask[a.attr] = om
            sat = Predicate._atom_sat(a, self._np[a.attr], self._isna[a.attr])
            obs_sat[id(a)] = sat
            hard_fail |= om & ~sat
            any_missing |= self._isna[a.attr]
            priors[id(a)] = float(sat[om].mean()) if om.any() else 0.0
        states = np.where(hard_fail, FALSE, TRUE).astype(object)
        p_phi = np.where(hard_fail, 0.0, 1.0)
        need = np.where(any_missing & ~hard_fail)[0]      # only these need inference
        for i in need:
            st = "pass"; prob = 1.0
            for atom in atoms:
                if obs_mask[atom.attr][i]:
                    if not obs_sat[id(atom)][i]:
                        st = "fail"; prob = 0.0; break
                    continue
                s, p = self._atom_for_tuple(atom, i, priors[id(atom)])
                if s == "fail":
                    st = "fail"; prob = 0.0; break
                if s == "uncertain":
                    st = "uncertain"; prob *= p
            if st == "fail":
                states[i] = FALSE; p_phi[i] = 0.0
            elif st == "uncertain":
                states[i] = UNKNOWN; p_phi[i] = prob
            else:
                states[i] = TRUE; p_phi[i] = 1.0
        return {"p_phi": p_phi, "pass_mask": states == TRUE,
                "fail_mask": states == FALSE, "uncertain_mask": states == UNKNOWN,
                "states": states}

    def eligible(self, cert, theta=0.5):
        return cert["pass_mask"] | (cert["uncertain_mask"] & (cert["p_phi"] >= theta))

    def _dom_cached(self, attr):
        if not hasattr(self, "_dom_cache"):
            self._dom_cache = {}
        if attr not in self._dom_cache:
            self._dom_cache[attr] = self.imv.domain_arrays(self.obs, attr)
        return self._dom_cache[attr]

    def certificate_fast(self, predicate, theta=0.5):
        """VECTORIZED certificate (no per-tuple Python loop). Bit-equivalent to certificate()
        under the soundness rule: only Sigma_ind (exact) hard-certifies; Sigma_dep stays uncertain
        with a probability. Uses cached per-attr domain arrays (one hash-join per attr)."""
        from mifa.predicates import Predicate
        n = len(self.obs)
        hard_fail = np.zeros(n, dtype=bool)
        any_unc = np.zeros(n, dtype=bool)
        p_phi = np.ones(n, dtype=np.float64)
        for at in predicate.atoms:
            a = at.attr
            isna = self._isna[a]; om = ~isna; vals = self._np[a]
            sat = Predicate._atom_sat(at, vals, isna)             # valid where observed
            prior = float(sat[om].mean()) if om.any() else 0.0
            hard_fail |= om & ~sat                                 # observed-false -> fail
            atom_p = np.where(om, sat.astype(float), prior)        # observed: 0/1; default prior
            atom_unc = isna.copy()                                 # missing tuples start uncertain
            dom = self._dom_cached(a)
            if dom is not None and dom["has"].any():
                has = isna & dom["has"]
                gid = dom["gid"]
                ng = len(dom["is_exact"])
                gidc = np.clip(gid, 0, max(ng - 1, 0))
                # COARSE-TO-FINE: exactness is now PER-GROUP (the finest firing level may be
                # Sigma_dep while a coarser one is Sigma_ind, or vice versa). is_exact[gid]
                # = the resolved group hard-certifies (Sigma_ind single-valued cat / Sigma_ind
                # numeric). Matches domain()'s ('exact',v) vs ('set',dist)/('interval',..).
                ex_grp = dom["is_exact"]
                ex_mask = has & (gid >= 0) & ex_grp[gidc]
                if self.dtypes[a] == "cat":
                    dists = dom["dists"]; gtop = dom["gtop"]
                    set_mask = has & ~ex_mask
                    if ex_mask.any():
                        exv = np.where(ex_mask, gtop[gidc], None)
                        oksat = Predicate._atom_sat(at, exv, ~ex_mask)
                        hard_fail |= ex_mask & ~oksat
                        atom_p = np.where(ex_mask, oksat.astype(float), atom_p)
                        atom_unc = np.where(ex_mask, False, atom_unc)
                    if dists:
                        pv = np.array([_atom_prob_cat(at, d) for d in dists])
                        atom_p = np.where(set_mask, pv[gidc], atom_p)
                else:
                    # numeric: exact groups (Sigma_ind) hard-certify on q50; the rest are
                    # Sigma_dep intervals -> uncertain with an overlap probability.
                    set_mask = has & ~ex_mask
                    if ex_mask.any():
                        oksat = Predicate._atom_sat(at, np.nan_to_num(dom["exact"], nan=0.0), ~ex_mask)
                        hard_fail |= ex_mask & ~oksat
                        atom_p = np.where(ex_mask, oksat.astype(float), atom_p)
                        atom_unc = np.where(ex_mask, False, atom_unc)
                    lo = dom["lo"]; hi = dom["hi"]; span = (hi - lo) + 1e-9
                    if at.op == "le":
                        p = (at.a - lo) / span
                    elif at.op == "ge":
                        p = (hi - at.a) / span
                    elif at.op == "between":
                        p = (np.minimum(hi, at.b) - np.maximum(lo, at.a)) / span
                    else:
                        p = np.full(n, prior)
                    atom_p = np.where(set_mask, np.clip(p, 0.0, 1.0), atom_p)
            p_phi = np.where(hard_fail, 0.0, p_phi * atom_p)
            any_unc |= atom_unc
        from mifa.predicates import TRUE as _T, FALSE as _F, UNKNOWN as _U
        states = np.full(n, _T, dtype=object)
        states[any_unc & ~hard_fail] = _U
        states[hard_fail] = _F
        p_phi = np.where(hard_fail, 0.0, p_phi)
        p_phi = np.where(states == _T, 1.0, p_phi)
        return {"p_phi": p_phi, "pass_mask": states == _T, "fail_mask": states == _F,
                "uncertain_mask": states == _U, "states": states}

    def certificate_batch(self, predicates, theta=0.5):
        """BATCH-VECTORIZED certificate over a LIST of predicates (a query batch).

        Returns a list of cert dicts, one per predicate, each BIT-IDENTICAL to
        certificate_fast(predicate). The win over a per-query loop: cross-query
        vectorization. Atoms across all B queries are grouped by (attr, op) so the
        expensive per-tuple work is shared:
          - per-attr cached domain arrays (_dom_cached) -> one lookup per attr, not per query;
          - categorical eq/in atom-prob: a (n_groups x n_distinct_values) probability matrix
            (gprob, precomputed once per attr) is gathered/summed across ALL queries at once,
            replacing certificate_fast's per-query Python `[_atom_prob_cat(at,d) for d in dists]`;
          - numeric le/ge/between dep-interval prob: the interval-overlap formula is broadcast
            across all queries' thresholds in one numpy expression;
          - observed/exact satisfaction is vectorized across queries sharing (attr, op).
        Object dtype is avoided in the hot path (int codes / float arrays).
        """
        from mifa.predicates import Predicate, TRUE as _T, FALSE as _F, UNKNOWN as _U
        B = len(predicates)
        n = len(self.obs)
        if B == 0:
            return []
        hard_fail = np.zeros((B, n), dtype=bool)
        any_unc = np.zeros((B, n), dtype=bool)
        p_phi = np.ones((B, n), dtype=np.float64)

        # Group atoms across ALL queries by (attr, op) so a whole group's per-tuple work is one
        # set of (G, n) numpy ops (G = #queries touching this attr+op) instead of G Python passes.
        by_ao = {}
        for qi, pred in enumerate(predicates):
            for at in pred.atoms:
                by_ao.setdefault((at.attr, at.op), []).append((qi, at))

        for (a, op), items in by_ao.items():
            qidx = np.fromiter((qi for qi, _ in items), dtype=np.int64, count=len(items))
            atoms = [at for _, at in items]
            G = len(atoms)
            isna = self._isna[a]; om = ~isna; vals = self._np[a]
            is_cat = self.dtypes[a] == "cat"
            dom = self._dom_cached(a)
            if not (dom is not None and dom.get("has") is not None and dom["has"].any()):
                dom = None
                has_dom = None
            else:
                has_dom = isna & dom["has"]

            # (G, n) observed-satisfaction + per-query prior + observed hard-fail, vectorized.
            sat = self._atom_sat_batch(op, atoms, vals, isna)        # (G, n) bool, valid where observed
            nobs = int(om.sum())
            prior = (sat[:, om].sum(axis=1) / nobs) if nobs else np.zeros(G)   # (G,)
            hf = om[None, :] & ~sat                                  # (G, n)
            atom_p = np.where(om[None, :], sat, prior[:, None])      # (G, n) float (bool->float ok)
            atom_unc = np.broadcast_to(isna, (G, n)).copy()

            if dom is not None:
                gid = dom["gid"]; ng = len(dom["is_exact"])
                gidc = np.clip(gid, 0, max(ng - 1, 0))
                # COARSE-TO-FINE: per-group exactness (see certificate_fast). The resolved
                # finest-firing group hard-certifies iff is_exact[gid].
                ex_grp = dom["is_exact"]
                ex_mask = has_dom & (gid >= 0) & ex_grp[gidc]   # (n,)
                set_mask = has_dom & ~ex_mask
                if is_cat:
                    if ex_mask.any():
                        # exact value per tuple -> evaluate the atom (G queries) against it
                        exv = dom["gtop"][gidc]                       # (n,) object, only valid on ex_mask
                        oksat = self._atom_sat_batch(op, atoms, exv, ~ex_mask)   # (G, n)
                        hf = hf | (ex_mask[None, :] & ~oksat)
                        atom_p = np.where(ex_mask[None, :], oksat, atom_p)
                        atom_unc[:, ex_mask] = False
                    if dom["dists"]:
                        # (G, n_groups) per-query group probs via the gprob matrix, then index by gidc
                        pv = self._cat_pv_batch(op, atoms, dom["gprob"], dom["val_index"])  # (G, n_groups)
                        atom_p = np.where(set_mask[None, :], pv[:, gidc], atom_p)
                else:
                    # numeric: Sigma_ind groups -> exact q50 (hard); rest -> Sigma_dep interval prob.
                    if ex_mask.any():
                        exq = np.nan_to_num(dom["exact"], nan=0.0)
                        oksat = self._atom_sat_batch(op, atoms, exq, ~ex_mask)
                        hf = hf | (ex_mask[None, :] & ~oksat)
                        atom_p = np.where(ex_mask[None, :], oksat, atom_p)
                        atom_unc[:, ex_mask] = False
                    lo = dom["lo"]; hi = dom["hi"]; span = (hi - lo) + 1e-9
                    if op in ("le", "ge", "between"):
                        thr_a = np.array([at.a for at in atoms], dtype=np.float64)[:, None]
                        if op == "le":
                            p = (thr_a - lo[None, :]) / span[None, :]
                        elif op == "ge":
                            p = (hi[None, :] - thr_a) / span[None, :]
                        else:
                            thr_b = np.array([at.b for at in atoms], dtype=np.float64)[:, None]
                            p = (np.minimum(hi[None, :], thr_b) - np.maximum(lo[None, :], thr_a)) / span[None, :]
                        atom_p = np.where(set_mask[None, :], np.clip(p, 0.0, 1.0), atom_p)
                    else:
                        atom_p = np.where(set_mask[None, :], prior[:, None], atom_p)

            # scatter into per-query rows. Fast path: when each query appears at most once in
            # this (attr,op) group (the single-atom-dominant case), qidx is unique -> the scatter
            # is a permutation, so direct fancy-index OR / multiply (buffered, fast). Only fall
            # back to the slow unbuffered ufunc.at when a query has two atoms with the SAME
            # (attr,op) (genuine duplicates needing accumulation).
            if len(np.unique(qidx)) == G:
                hard_fail[qidx] |= hf
                p_phi[qidx] *= atom_p
                any_unc[qidx] |= atom_unc
            else:
                np.logical_or.at(hard_fail, qidx, hf)
                np.multiply.at(p_phi, qidx, atom_p)
                np.logical_or.at(any_unc, qidx, atom_unc)

        # zero p_phi on hard-fail (it must be 0, not the product) BEFORE building states.
        p_phi = np.where(hard_fail, 0.0, p_phi)

        # Build per-query masks (booleans, no per-query object np.full). states is derived once
        # via a 3-element code->string lookup gather (cheaper than np.full(n, _T, object)+assigns).
        fail_mask = hard_fail
        unc_mask = any_unc & ~hard_fail
        pass_mask = ~hard_fail & ~any_unc
        p_phi = np.where(pass_mask, 1.0, p_phi)            # TRUE tuples -> p_phi = 1.0
        codes = np.where(fail_mask, 0, np.where(unc_mask, 1, 2)).astype(np.intp)   # (B, n)
        lut = np.array([_F, _U, _T], dtype=object)
        states_all = lut[codes]                            # (B, n) object, one gather per row block
        out = []
        for qi in range(B):
            out.append({"p_phi": p_phi[qi], "pass_mask": pass_mask[qi], "fail_mask": fail_mask[qi],
                        "uncertain_mask": unc_mask[qi], "states": states_all[qi]})
        return out

    @staticmethod
    def _atom_sat_batch(op, atoms, vals, missing):
        """Vectorized satisfaction of G atoms (all same op) over n values -> (G, n) bool.
        Result undefined where missing (treated as not-satisfied, as in Predicate._atom_sat)."""
        G = len(atoms); n = len(vals)
        valid = ~missing
        out = np.zeros((G, n), dtype=bool)
        if not valid.any():
            return out
        if op in ("le", "ge", "between"):
            v = np.asarray(vals, dtype=float)
            thr_a = np.array([at.a for at in atoms], dtype=np.float64)[:, None]
            if op == "le":
                cmp = v[None, :] <= thr_a
            elif op == "ge":
                cmp = v[None, :] >= thr_a
            else:
                thr_b = np.array([at.b for at in atoms], dtype=np.float64)[:, None]
                cmp = (v[None, :] >= thr_a) & (v[None, :] <= thr_b)
            return cmp & valid[None, :]
        if op == "eq":
            vobj = np.asarray(vals, dtype=object)
            for gi, at in enumerate(atoms):
                row = (vobj == at.a) & valid
                out[gi] = row
            return out
        if op == "in":
            vobj = np.asarray(vals, dtype=object)
            vi = np.where(valid)[0]
            for gi, at in enumerate(atoms):
                S = set(at.a)
                for i in vi:
                    out[gi, i] = vobj[i] in S
            return out
        raise ValueError(op)

    @staticmethod
    def _cat_pv_batch(op, atoms, gprob, val_index):
        """Per-group P(atom satisfied) for G categorical eq/in atoms -> (G, n_groups), via the
        precomputed gprob matrix (column gather/sum). Matches [_atom_prob_cat(at,d) for d]."""
        G = len(atoms); ng = gprob.shape[0]
        out = np.zeros((G, ng), dtype=np.float64)
        if op == "eq":
            cols = np.array([val_index.get(at.a, -1) for at in atoms], dtype=np.int64)
            ok = cols >= 0
            if ok.any():
                out[ok] = gprob[:, cols[ok]].T
            return out
        if op == "in":
            for gi, at in enumerate(atoms):
                cs = [val_index[v] for v in dict.fromkeys(at.a) if v in val_index]
                if cs:
                    out[gi] = gprob[:, cs].sum(axis=1)
            return out
        raise ValueError(f"cat atom op {op}")

    @staticmethod
    def _cat_atom_pv(atom, gprob, val_index):
        """Per-group P(atom satisfied) for a categorical eq/in atom, vectorized over groups
        via the precomputed gprob matrix. Equivalent to [_atom_prob_cat(atom, d) for d in dists]
        but a numpy column-gather/sum instead of a Python per-group loop."""
        if atom.op == "eq":
            j = val_index.get(atom.a)
            if j is None:
                return np.zeros(gprob.shape[0], dtype=np.float64)
            return gprob[:, j]
        if atom.op == "in":
            cols = [val_index[v] for v in dict.fromkeys(atom.a) if v in val_index]
            if not cols:
                return np.zeros(gprob.shape[0], dtype=np.float64)
            return gprob[:, cols].sum(axis=1)
        raise ValueError(f"cat atom op {atom.op}")

    def p_phi_tuple(self, t_id, predicate):
        """Indexed per-tuple p_phi for ONE tuple via the I_mv group lookup -- O(#atoms)
        index probes, NOT a full observed-pool scan. This is the paper's fine refinement
        (a finer I_mv probe), the sublinear alternative to embedding-kNN refinement."""
        from mifa.predicates import Predicate
        p = 1.0
        for atom in predicate.atoms:
            a = atom.attr
            if not self._isna[a][t_id]:
                if atom.eval_value(self._np[a][t_id]) != TRUE:
                    return 0.0
                continue
            ckey = (a, atom.op, getattr(atom, "a", None),
                    tuple(atom.a) if isinstance(getattr(atom, "a", None), (list, set)) else getattr(atom, "b", None))
            cache = getattr(self, "_prior_cache", None)
            if cache is None:
                cache = self._prior_cache = {}
            prior = cache.get(ckey)
            if prior is None:                          # compute once per atom, then O(1)
                om = ~self._isna[a]
                sat = Predicate._atom_sat(atom, self._np[a], self._isna[a])
                prior = float(sat[om].mean()) if om.any() else 0.0
                cache[ckey] = prior
            s, pp = self._atom_for_tuple(atom, t_id, prior)
            if s == "fail":
                return 0.0
            if s == "uncertain":
                p *= pp
        return p
