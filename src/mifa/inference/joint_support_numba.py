"""Fused Numba context counting for protocol-v2 joint residual support.

The reference engine is intentionally simple, but it repeatedly enters pandas
for every unresolved target tuple. This kernel fuses body/cell matching,
complete-support counting, residual satisfaction counting, shrinkage, and
context selection into one parallel native-code pass per residual pattern.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


GLOBAL = 0
CELL = 1
BODY = 2
BODY_CELL = 3
ABSTAIN = -1


@njit(cache=True, fastmath=False)
def _raw_probability(count, joint_plus, atom_plus, mode):
    if count == 0:
        return np.nan, 0
    if mode == 0:
        return joint_plus / count, joint_plus
    value = 1.0 if mode == 1 else np.inf
    for atom in range(atom_plus.shape[0]):
        probability = atom_plus[atom] / count
        if mode == 1:
            value *= probability
        elif probability < value:
            value = probability
    return value, int(np.rint(value * count))


@njit(cache=True, fastmath=False)
def _shrink(raw, count, broader, kappa):
    if not np.isfinite(raw):
        return broader
    if np.isfinite(broader):
        return (count * raw + kappa * broader) / (count + kappa)
    return raw


@njit(parallel=True, cache=True, fastmath=False)
def joint_context_kernel(complete, joint_satisfied, atom_satisfied,
                         body_support_codes, body_target_codes,
                         support_cells, target_cells, use_cells,
                         mode, min_support, kappa,
                         global_p, global_n, global_nplus):
    """Return selected p/count/context for each unresolved target row."""
    n_targets = body_target_codes.shape[1]
    n_support = complete.shape[0]
    n_body = body_support_codes.shape[0]
    n_atoms = atom_satisfied.shape[0]
    selected_p = np.zeros(n_targets, dtype=np.float64)
    selected_n = np.zeros(n_targets, dtype=np.int64)
    selected_nplus = np.zeros(n_targets, dtype=np.int64)
    selected_context = np.full(n_targets, ABSTAIN, dtype=np.int8)

    for target in prange(n_targets):
        visible_body = 0
        for body in range(n_body):
            if body_target_codes[body, target] >= 0:
                visible_body += 1

        body_n = 0
        body_joint_plus = 0
        body_atom_plus = np.zeros(n_atoms, dtype=np.int64)
        local_n = 0
        local_joint_plus = 0
        local_atom_plus = np.zeros(n_atoms, dtype=np.int64)

        for row in range(n_support):
            if not complete[row]:
                continue
            body_match = True
            if visible_body:
                for body in range(n_body):
                    target_code = body_target_codes[body, target]
                    if target_code >= 0 and body_support_codes[body, row] != target_code:
                        body_match = False
                        break
            if visible_body and body_match:
                body_n += 1
                if mode == 0:
                    body_joint_plus += int(joint_satisfied[row])
                else:
                    for atom in range(n_atoms):
                        body_atom_plus[atom] += int(atom_satisfied[atom, row])

            local_match = use_cells and support_cells[row] == target_cells[target]
            if visible_body:
                local_match = local_match and body_match
            if local_match:
                local_n += 1
                if mode == 0:
                    local_joint_plus += int(joint_satisfied[row])
                else:
                    for atom in range(n_atoms):
                        local_atom_plus[atom] += int(atom_satisfied[atom, row])

        body_raw, body_nplus = _raw_probability(
            body_n, body_joint_plus, body_atom_plus, mode)
        local_raw, local_nplus = _raw_probability(
            local_n, local_joint_plus, local_atom_plus, mode)

        if visible_body:
            body_p = _shrink(body_raw, body_n, global_p, kappa)
            if use_cells:
                local_p = _shrink(local_raw, local_n, body_p, kappa)
                if local_n >= min_support and np.isfinite(local_p):
                    selected_p[target] = local_p
                    selected_n[target] = local_n
                    selected_nplus[target] = local_nplus
                    selected_context[target] = BODY_CELL
                    continue
            if body_n >= min_support and np.isfinite(body_p):
                selected_p[target] = body_p
                selected_n[target] = body_n
                selected_nplus[target] = body_nplus
                selected_context[target] = BODY
                continue
        elif use_cells:
            local_p = _shrink(local_raw, local_n, global_p, kappa)
            if local_n >= min_support and np.isfinite(local_p):
                selected_p[target] = local_p
                selected_n[target] = local_n
                selected_nplus[target] = local_nplus
                selected_context[target] = CELL
                continue

        if global_n >= min_support and np.isfinite(global_p):
            selected_p[target] = global_p
            selected_n[target] = global_n
            selected_nplus[target] = global_nplus
            selected_context[target] = GLOBAL

    return selected_p, selected_n, selected_nplus, selected_context


@njit(cache=True, fastmath=False)
def _group_probability(count, joint_plus, atom_plus, group, mode):
    if count == 0:
        return np.nan, 0
    if mode == 0:
        return joint_plus / count, joint_plus
    value = 1.0 if mode == 1 else np.inf
    for atom in range(atom_plus.shape[0]):
        probability = atom_plus[atom, group] / count
        if mode == 1:
            value *= probability
        elif probability < value:
            value = probability
    return value, int(np.rint(value * count))


@njit(cache=True, fastmath=False)
def joint_grouped_context_kernel(
        complete, joint_satisfied, atom_satisfied,
        support_body_groups, target_body_groups, n_body_groups,
        support_cell_groups, target_cell_groups, n_cell_groups,
        support_body_cell_groups, target_body_cell_groups, n_body_cell_groups,
        use_body, use_cells, mode, min_support, kappa):
    """Aggregate each context once, then resolve all targets in O(S + T).

    ``joint_context_kernel`` is the readable fused baseline but scans all S
    support rows for every unresolved target.  Main-table workloads have fixed,
    observed body attributes, so context membership can be encoded as compact
    group ids.  This kernel fuses the one-pass group aggregation and target
    selection while preserving the exact shrinkage/backoff semantics.
    """
    n_atoms = atom_satisfied.shape[0]
    global_n = 0
    global_joint_plus = 0
    global_atom_plus = np.zeros(n_atoms, dtype=np.int64)
    body_n = np.zeros(n_body_groups, dtype=np.int64)
    body_joint_plus = np.zeros(n_body_groups, dtype=np.int64)
    body_atom_plus = np.zeros((n_atoms, n_body_groups), dtype=np.int64)
    cell_n = np.zeros(n_cell_groups, dtype=np.int64)
    cell_joint_plus = np.zeros(n_cell_groups, dtype=np.int64)
    cell_atom_plus = np.zeros((n_atoms, n_cell_groups), dtype=np.int64)
    body_cell_n = np.zeros(n_body_cell_groups, dtype=np.int64)
    body_cell_joint_plus = np.zeros(n_body_cell_groups, dtype=np.int64)
    body_cell_atom_plus = np.zeros(
        (n_atoms, n_body_cell_groups), dtype=np.int64)

    for row in range(complete.shape[0]):
        if not complete[row]:
            continue
        global_n += 1
        global_joint_plus += int(joint_satisfied[row])
        for atom in range(n_atoms):
            global_atom_plus[atom] += int(atom_satisfied[atom, row])
        if use_body:
            group = support_body_groups[row]
            body_n[group] += 1
            body_joint_plus[group] += int(joint_satisfied[row])
            for atom in range(n_atoms):
                body_atom_plus[atom, group] += int(atom_satisfied[atom, row])
        if use_cells:
            group = support_cell_groups[row]
            cell_n[group] += 1
            cell_joint_plus[group] += int(joint_satisfied[row])
            for atom in range(n_atoms):
                cell_atom_plus[atom, group] += int(atom_satisfied[atom, row])
        if use_body and use_cells:
            group = support_body_cell_groups[row]
            body_cell_n[group] += 1
            body_cell_joint_plus[group] += int(joint_satisfied[row])
            for atom in range(n_atoms):
                body_cell_atom_plus[atom, group] += int(atom_satisfied[atom, row])

    if global_n == 0:
        global_p = np.nan
        global_nplus = 0
    elif mode == 0:
        global_p = global_joint_plus / global_n
        global_nplus = global_joint_plus
    else:
        global_p = 1.0 if mode == 1 else np.inf
        for atom in range(n_atoms):
            probability = global_atom_plus[atom] / global_n
            if mode == 1:
                global_p *= probability
            elif probability < global_p:
                global_p = probability
        global_nplus = int(np.rint(global_p * global_n))

    n_targets = target_body_groups.shape[0]
    selected_p = np.zeros(n_targets, dtype=np.float64)
    selected_n = np.zeros(n_targets, dtype=np.int64)
    selected_nplus = np.zeros(n_targets, dtype=np.int64)
    selected_context = np.full(n_targets, ABSTAIN, dtype=np.int8)
    for target in range(n_targets):
        if use_body:
            body_group = target_body_groups[target]
            body_raw, body_nplus = _group_probability(
                body_n[body_group], body_joint_plus[body_group],
                body_atom_plus, body_group, mode)
            body_p = _shrink(body_raw, body_n[body_group], global_p, kappa)
            if use_cells:
                local_group = target_body_cell_groups[target]
                local_raw, local_nplus = _group_probability(
                    body_cell_n[local_group], body_cell_joint_plus[local_group],
                    body_cell_atom_plus, local_group, mode)
                local_p = _shrink(
                    local_raw, body_cell_n[local_group], body_p, kappa)
                if body_cell_n[local_group] >= min_support and np.isfinite(local_p):
                    selected_p[target] = local_p
                    selected_n[target] = body_cell_n[local_group]
                    selected_nplus[target] = local_nplus
                    selected_context[target] = BODY_CELL
                    continue
            if body_n[body_group] >= min_support and np.isfinite(body_p):
                selected_p[target] = body_p
                selected_n[target] = body_n[body_group]
                selected_nplus[target] = body_nplus
                selected_context[target] = BODY
                continue
        elif use_cells:
            cell_group = target_cell_groups[target]
            local_raw, local_nplus = _group_probability(
                cell_n[cell_group], cell_joint_plus[cell_group],
                cell_atom_plus, cell_group, mode)
            local_p = _shrink(local_raw, cell_n[cell_group], global_p, kappa)
            if cell_n[cell_group] >= min_support and np.isfinite(local_p):
                selected_p[target] = local_p
                selected_n[target] = cell_n[cell_group]
                selected_nplus[target] = local_nplus
                selected_context[target] = CELL
                continue
        if global_n >= min_support and np.isfinite(global_p):
            selected_p[target] = global_p
            selected_n[target] = global_n
            selected_nplus[target] = global_nplus
            selected_context[target] = GLOBAL
    return selected_p, selected_n, selected_nplus, selected_context
