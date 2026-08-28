"""MIFA-minimal certificate engine (Phase 2a).

Given a predicate phi, the observed (masked) table, and a coarse cell index, this
computes per-tuple satisfaction probability p_phi(t) and a 3-valued certificate:

  pass (TRUE)   : all relevant atoms observed AND satisfied            -> p_phi = 1
  fail (FALSE)  : some relevant atom observed AND violated             -> p_phi = 0
  uncertain     : >=1 relevant atom missing, not hard-failed           -> p_phi in (0,1)

A tuple is ELIGIBLE iff it passes, or it is uncertain with p_phi >= theta.
Eligibility decides candidacy; embedding distance decides ranking (done by search).

This is the *naive coarse* inference: per-atom probability for a missing attribute
is the empirical satisfaction rate of that atom among observed values in the
tuple's embedding cell. Atoms are combined assuming conditional independence
given the cell (a deliberate Phase-2a simplification; rules come in Phase 2c).
"""
from __future__ import annotations

import numpy as np

from mifa.predicates import Predicate


def certificate(spec, observed_table, predicate: Predicate, cell_index):
    """Return dict with p_phi (n,), pass_mask, fail_mask, uncertain_mask (bool n,)."""
    n = len(observed_table)
    hard_fail = np.zeros(n, dtype=bool)
    any_missing = np.zeros(n, dtype=bool)
    # product of per-atom probabilities (1 where observed&satisfied, p_cell where missing)
    p_prod = np.ones(n, dtype=np.float64)

    for at in predicate.atoms:
        col = observed_table[at.attr]
        observed = ~col.isna().to_numpy()
        vals = col.to_numpy()
        sat_obs = Predicate._atom_sat(at, vals, ~observed)   # valid where observed

        # observed & violated -> hard fail
        hard_fail |= observed & (~sat_obs)
        # observed & satisfied -> factor 1 (no-op). missing -> probabilistic factor
        missing = ~observed
        any_missing |= missing
        if missing.any():
            _, p_tuple = cell_index.cell_atom_prob(sat_obs, observed)
            p_prod = np.where(missing, p_prod * p_tuple, p_prod)

    p_phi = np.where(hard_fail, 0.0, p_prod)
    pass_mask = (~hard_fail) & (~any_missing)      # all observed & satisfied
    fail_mask = hard_fail
    uncertain_mask = (~hard_fail) & any_missing
    p_phi = np.where(pass_mask, 1.0, p_phi)
    return {
        "p_phi": p_phi,
        "pass_mask": pass_mask,
        "fail_mask": fail_mask,
        "uncertain_mask": uncertain_mask,
    }


def eligible_mask(cert, theta):
    """Eligible iff pass, or uncertain with p_phi >= theta."""
    return cert["pass_mask"] | (cert["uncertain_mask"] & (cert["p_phi"] >= theta))
