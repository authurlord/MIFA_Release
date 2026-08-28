#!/usr/bin/env python3
"""Production numba-CUDA numeric certificate (fast-prior), lifted from scripts/cert_headtohead.py.

This is the head-to-head WINNER for NUMERIC single-atom predicates (le/ge/between): on full
arxiv (n=100000, 4096-d) it reached ~25k q/s end-to-end vs ~639 q/s for the torch-GPU resident
path with the same searchsorted fast-prior, because the @cuda.jit kernel avoids the per-batch
torch tensor-building overhead. Eligibility is BIT-EXACT to
``rule_engine.certificate_batch`` + ``eligible(theta)`` for single numeric atoms.

Design (mirrors GPUResidentNumericCert math, on device, float64 throughout):
  - per-attr STATIC arrays (isna / value / dep-interval lo,hi / Sigma_ind exact / has_dom) are
    uploaded to the device ONCE and kept RESIDENT (``CudaNumericCert._ensure``).
  - per batch the only host->device traffic is the tiny (B,) threshold + prior vectors.
  - the prior is computed in O(B log n) via searchsorted on the sorted observed values
    (the "fast-prior") instead of a per-query O(n) scan (the real arxiv CPU wall).
  - one @cuda.jit thread per (query b, tuple i) writes the (B,n) int8 eligibility on device;
    the caller converts it to a torch bool tensor zero-copy (``torch.as_tensor(elig, device=...)``)
    so the cert stays on-GPU and feeds the fused exact-over-eligible top-k with no round-trip.

CORRECTNESS: every probability / value-vs-threshold / p>=theta / prior>=theta comparison is in
float64. arxiv's update_date is a large integer date-code (~2e7); fp32 has only ~24 mantissa bits
so a boundary (value==threshold, p==theta) could FLIP vs the fp64 CPU certificate -> recall drift.

SCOPE: single numeric atom (le/ge/between). Predicates outside that scope must be routed by the
caller to ``cert_numba.certificate_batch_numba`` (categorical) or ``certificate_batch`` (other).
``supports(predicate)`` reports membership.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import cuda
    HAVE_CUDA = cuda.is_available()
except Exception:                                          # pragma: no cover
    cuda = None
    HAVE_CUDA = False


_OP = {"le": 0, "ge": 1, "between": 2}


# ===========================================================================  device kernel
if HAVE_CUDA:

    @cuda.jit
    def _elig_num_cuda(n, B, op_code, isna, vals, has_dom, lo, hi, exact, ind,
                       a_thr, b_thr, prior, theta, elig):
        """One thread per (query b, tuple i). Writes elig (B,n) int8 on device. float64 math.

        op_code: 0=le, 1=ge, 2=between. Mirrors certificate_batch eligibility exactly:
          observed             -> hard pass iff atom(value)
          missing, dep-rule    -> p = clip(overlap(thr,[lo,hi]),0,1); eligible iff p>=theta
          missing, ind-rule    -> hard pass iff atom(exact)              (only if ind==1)
          missing, no rule     -> eligible iff prior>=theta
        """
        b, i = cuda.grid(2)
        if b >= B or i >= n:
            return
        a = a_thr[b]
        bb = b_thr[b]
        if not isna[i]:
            v = vals[i]
            if op_code == 0:
                ok = v <= a
            elif op_code == 1:
                ok = v >= a
            else:
                ok = (v >= a) and (v <= bb)
            elig[b, i] = 1 if ok else 0
            return
        if has_dom[i]:
            if ind[i] == 1:                     # PER-TUPLE exactness (coarse-to-fine)
                e = exact[i]
                if op_code == 0:
                    ok = e <= a
                elif op_code == 1:
                    ok = e >= a
                else:
                    ok = (e >= a) and (e <= bb)
                elig[b, i] = 1 if ok else 0
                return
            l = lo[i]; h = hi[i]
            span = (h - l) + 1e-9
            if op_code == 0:
                p = (a - l) / span
            elif op_code == 1:
                p = (h - a) / span
            else:
                mn = h if h < bb else bb
                mx = l if l > a else a
                p = (mn - mx) / span
            if p < 0.0:
                p = 0.0
            elif p > 1.0:
                p = 1.0
            elig[b, i] = 1 if p >= theta else 0
            return
        elig[b, i] = 1 if prior[b] >= theta else 0


# ===========================================================================  certificate
class CudaNumericCert:
    """Resident numba-CUDA numeric-atom certificate (fast-prior). Covers single numeric atoms
    (le/ge/between). Use ``supports`` to gate, then ``eligible_batch_cuda`` per batch."""

    def __init__(self, eng, spec):
        if not HAVE_CUDA:
            raise RuntimeError("CudaNumericCert requires an available CUDA device (numba.cuda).")
        self.eng = eng
        self.dtypes = spec.dtypes
        self._dev_static = {}        # attr -> resident device arrays (uploaded once)
        self._sorted_obs_cache = {}  # attr -> (sorted observed values, count)

    # ----------------------------------------------------------------- scope
    def supports(self, predicate):
        ats = predicate.atoms
        if len(ats) != 1:
            return False
        at = ats[0]
        return self.dtypes.get(at.attr) == "num" and at.op in ("le", "ge", "between")

    # ----------------------------------------------------------------- static device arrays
    def _host_static(self, attr):
        """Per-attr STATIC host arrays (fp64) matching GPUResidentNumericCert._ensure exactly."""
        eng = self.eng
        isna = eng._isna[attr]
        om = ~isna
        vals = eng._np[attr].astype(np.float64)
        vals_f = np.nan_to_num(vals, nan=0.0)
        dom = eng._dom_cached(attr)
        n = len(isna)
        lo = np.zeros(n, np.float64); hi = np.zeros(n, np.float64)
        exact = np.zeros(n, np.float64)
        ind_arr = np.zeros(n, dtype=np.int8); has_dom = np.zeros(n, bool)
        if dom is not None and dom.get("has") is not None and dom["has"].any():
            has_dom = (isna & dom["has"])
            # COARSE-TO-FINE: exactness is PER-TUPLE (ind_tuple). Exact tuples carry q50; dep
            # tuples carry the [lo,hi] interval. A single attr may now have BOTH (different
            # finest-firing levels), so populate both arrays and switch per tuple in the kernel.
            ind_tuple = dom["ind_tuple"]
            ind_arr = ind_tuple.astype(np.int8)
            exact = np.nan_to_num(dom["exact"], nan=0.0).astype(np.float64)
            lo_raw = np.nan_to_num(dom["lo"], nan=0.0).astype(np.float64)
            hi_raw = np.nan_to_num(dom["hi"], nan=0.0).astype(np.float64)
            dep_dom = has_dom & ~ind_tuple
            assert np.isfinite(dom["lo"][dep_dom]).all() and np.isfinite(dom["hi"][dep_dom]).all(), \
                f"non-finite dep interval for attr {attr} where has_dom & dep"
            lo = lo_raw; hi = hi_raw
        return dict(isna=isna.astype(np.bool_), vals=vals_f, has_dom=has_dom.astype(np.bool_),
                    lo=lo, hi=hi, exact=exact, ind=ind_arr, n=n, om=om, vals_raw=vals)

    def _ensure(self, attr):
        """Upload static arrays for `attr` to the device ONCE (resident). Idempotent."""
        if attr in self._dev_static:
            return self._dev_static[attr]
        st = self._host_static(attr)
        d = dict(
            isna=cuda.to_device(st["isna"]),
            vals=cuda.to_device(st["vals"]),
            has_dom=cuda.to_device(st["has_dom"]),
            lo=cuda.to_device(st["lo"]),
            hi=cuda.to_device(st["hi"]),
            exact=cuda.to_device(st["exact"]),
            ind=cuda.to_device(st["ind"]),       # PER-TUPLE int8 exactness array (coarse-to-fine)
            n=st["n"],
        )
        self._dev_static[attr] = d
        # warm the sorted-observed cache too (used by the fast prior)
        self._sorted_obs(attr, st)
        return d

    def warmup(self, attr):
        """Upload statics + JIT-compile the kernel for `attr` (so first timed batch pays neither)."""
        d = self._ensure(attr)
        # one tiny dummy launch to trigger numba JIT compilation
        a_d = cuda.to_device(np.zeros(1, np.float64))
        b_d = cuda.to_device(np.zeros(1, np.float64))
        pr_d = cuda.to_device(np.zeros(1, np.float64))
        elig = cuda.device_array((1, d["n"]), dtype=np.int8)
        tpb = (8, 64)
        bpg = ((1 + tpb[0] - 1) // tpb[0], (d["n"] + tpb[1] - 1) // tpb[1])
        _elig_num_cuda[bpg, tpb](d["n"], 1, 0, d["isna"], d["vals"], d["has_dom"],
                                 d["lo"], d["hi"], d["exact"], d["ind"],
                                 a_d, b_d, pr_d, 0.5, elig)
        cuda.synchronize()

    # ----------------------------------------------------------------- fast prior
    def _sorted_obs(self, attr, st=None):
        """Cache SORTED observed values for `attr` -> O(B log n) batched prior via searchsorted."""
        if attr in self._sorted_obs_cache:
            return self._sorted_obs_cache[attr]
        if st is None:
            st = self._host_static(attr)
        vals = st["vals_raw"]
        obs = vals[st["om"]]
        obs = obs[np.isfinite(obs)]
        obs = np.sort(obs.astype(np.float64))
        self._sorted_obs_cache[attr] = (obs, len(obs))
        return self._sorted_obs_cache[attr]

    def _query_thr_prior_fast(self, attr, op, predicates):
        """Vectorized per-batch (a_thr, b_thr, prior). prior via searchsorted on sorted observed
        values -> bit-equal to the fraction-of-observed prior, O(B log n) not O(B n)."""
        obs, nobs = self._sorted_obs(attr)
        a_thr = np.array([float(p.atoms[0].a) for p in predicates], np.float64)
        if op == "between":
            b_thr = np.array([float(p.atoms[0].b) for p in predicates], np.float64)
        else:
            b_thr = np.zeros(len(predicates), np.float64)
        if nobs == 0:
            return a_thr, b_thr, np.zeros(len(predicates), np.float64)
        if op == "le":
            cnt = np.searchsorted(obs, a_thr, side="right")
            prior = cnt / nobs
        elif op == "ge":
            cnt = nobs - np.searchsorted(obs, a_thr, side="left")
            prior = cnt / nobs
        else:  # between [a,b]
            hi = np.searchsorted(obs, b_thr, side="right")
            lo = np.searchsorted(obs, a_thr, side="left")
            prior = (hi - lo) / nobs
        return a_thr, b_thr, prior.astype(np.float64)

    # ----------------------------------------------------------------- eligibility
    def eligible_batch_cuda(self, predicates, theta=0.5):
        """Return a (B, n) int8 numba-cuda DEVICE array of eligibility for a batch of single
        numeric-atom predicates (the batch may mix attrs/ops; grouped per (attr, op)).

        The result stays on-device; the caller should wrap it via ``torch.as_tensor(elig,
        device='cuda').to(torch.bool)`` (zero-copy) to feed the fused top-k."""
        attr0 = predicates[0].atoms[0].attr
        d0 = self._ensure(attr0)
        B = len(predicates)
        # All-supported batches usually share one attr; still handle a mixed batch correctly.
        n = d0["n"]
        elig = cuda.device_array((B, n), dtype=np.int8)
        groups = {}
        for j, p in enumerate(predicates):
            at = p.atoms[0]
            groups.setdefault((at.attr, at.op), []).append(j)
        tpb = (8, 64)
        # Fast path: single (attr, op) group -> one kernel writes the whole (B,n) directly.
        if len(groups) == 1:
            (attr, op), js = next(iter(groups.items()))
            d = self._ensure(attr)
            a_thr, b_thr, prior = self._query_thr_prior_fast(attr, op, predicates)
            a_d = cuda.to_device(a_thr); b_d = cuda.to_device(b_thr); pr_d = cuda.to_device(prior)
            bpg = ((B + tpb[0] - 1) // tpb[0], (n + tpb[1] - 1) // tpb[1])
            _elig_num_cuda[bpg, tpb](n, B, _OP[op], d["isna"], d["vals"], d["has_dom"],
                                     d["lo"], d["hi"], d["exact"], d["ind"],
                                     a_d, b_d, pr_d, float(theta), elig)
            return elig
        # Mixed batch: kernel per group into a (g,n) buffer, then scatter rows back.
        for (attr, op), js in groups.items():
            d = self._ensure(attr)
            sub = [predicates[j] for j in js]
            a_thr, b_thr, prior = self._query_thr_prior_fast(attr, op, sub)
            g = len(js)
            sub_elig = cuda.device_array((g, n), dtype=np.int8)
            a_d = cuda.to_device(a_thr); b_d = cuda.to_device(b_thr); pr_d = cuda.to_device(prior)
            bpg = ((g + tpb[0] - 1) // tpb[0], (n + tpb[1] - 1) // tpb[1])
            _elig_num_cuda[bpg, tpb](n, g, _OP[op], d["isna"], d["vals"], d["has_dom"],
                                     d["lo"], d["hi"], d["exact"], d["ind"],
                                     a_d, b_d, pr_d, float(theta), sub_elig)
            host = sub_elig.copy_to_host()
            for jj, j in enumerate(js):
                elig[j].copy_to_device(host[jj])
        return elig
