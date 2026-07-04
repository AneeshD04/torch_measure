# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Private helpers for balanced, fully crossed multi-facet G-theory designs."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _effect_key(axes_frozenset: frozenset[int], facet_list: list[str]):
    """Convert frozenset of axis indices to a component key (str or tuple)."""
    names = tuple(facet_list[i] for i in sorted(axes_frozenset))
    return names[0] if len(names) == 1 else names


def _validate_design(df: pd.DataFrame, facet_cols: list[str], response_col: str):
    """Validate a balanced, fully crossed multi-facet design.

    Returns
    -------
    n_levels : dict[str, int]
    n_reps : int
    sorted_levels : dict[str, list]
    """
    import pandas as pd

    if len(facet_cols) != len(set(facet_cols)):
        seen = set()
        dupes = [f for f in facet_cols if f in seen or seen.add(f)]
        raise ValueError(f"Duplicate facet_cols: {dupes}.")

    missing = (set(facet_cols) | {response_col}) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}.")

    if not pd.api.types.is_numeric_dtype(df[response_col]):
        raise ValueError(f"{response_col!r} column must be numeric.")

    n_levels: dict[str, int] = {}
    sorted_levels: dict[str, list] = {}
    for f in facet_cols:
        levs = sorted(df[f].unique(), key=str)
        if len(levs) < 2:
            raise ValueError(f"Facet {f!r} needs at least 2 levels; got {len(levs)}.")
        n_levels[f] = len(levs)
        sorted_levels[f] = levs

    expected_cells = 1
    for n in n_levels.values():
        expected_cells *= n

    cell_counts = df.groupby(facet_cols, observed=True)[response_col].count()

    if len(cell_counts) < expected_cells:
        raise ValueError(f"Design is not fully crossed: {len(cell_counts)}/{expected_cells} cells observed.")

    counts_arr = cell_counts.values
    cmin, cmax = int(counts_arr.min()), int(counts_arr.max())
    if cmin != cmax:
        raise ValueError(f"Unequal replication: cell counts range from {cmin} to {cmax}.")

    return n_levels, int(counts_arr[0]), sorted_levels


def _parse_interactions(interactions, facet_list: list[str]):
    """Parse *interactions* into a set of frozensets of axis indices.

    Main effects are always included. Returns ``None`` when ``"all"``.
    """
    if interactions == "all":
        return None

    facet_to_axis = {f: i for i, f in enumerate(facet_list)}
    selected: set[frozenset[int]] = set()

    for i in range(len(facet_list)):
        selected.add(frozenset([i]))

    for inter in interactions:
        if isinstance(inter, str):
            if inter not in facet_to_axis:
                raise ValueError(f"Unknown facet in interactions: {inter!r}.")
            selected.add(frozenset([facet_to_axis[inter]]))
        else:
            for f in inter:
                if f not in facet_to_axis:
                    raise ValueError(f"Unknown facet in interactions: {f!r}.")
            selected.add(frozenset(facet_to_axis[f] for f in inter))

    return selected


def _build_cell_means(
    df: pd.DataFrame,
    facet_list: list[str],
    response_col: str,
    n_levels: dict[str, int],
    sorted_levels: dict[str, list],
) -> np.ndarray:
    """Return an ndarray of cell means with shape ``(n_f1, n_f2, ...)``."""
    cell_means_series = df.groupby(facet_list, observed=True)[response_col].mean()
    shape = tuple(n_levels[f] for f in facet_list)
    level_to_idx = {f: {lev: i for i, lev in enumerate(sorted_levels[f])} for f in facet_list}

    means = np.empty(shape, dtype=float)
    for idx_tuple, val in cell_means_series.items():
        if not isinstance(idx_tuple, tuple):
            idx_tuple = (idx_tuple,)
        arr_idx = tuple(level_to_idx[f][idx_tuple[j]] for j, f in enumerate(facet_list))
        means[arr_idx] = val

    return means


def _compute_ms_error(
    df: pd.DataFrame,
    facet_list: list[str],
    response_col: str,
    n_levels: dict[str, int],
    n_reps: int,
) -> float:
    """Within-cell mean-square error (0 when *n_reps* == 1)."""
    if n_reps <= 1:
        return 0.0
    cell_mean_col = df.groupby(facet_list, observed=True)[response_col].transform("mean")
    ss_e = float(((df[response_col] - cell_mean_col) ** 2).sum())
    total_cells = int(np.prod(list(n_levels.values())))
    df_e = total_cells * (n_reps - 1)
    return ss_e / df_e


def _anova_components(
    means_array: np.ndarray,
    facet_list: list[str],
    n_levels: dict[str, int],
    n_reps: int,
    ms_e: float,
    selected: set[frozenset[int]] | None,
):
    """Balanced ANOVA method-of-moments for variance components.

    When ``n_reps == 1``, residual is confounded with the full interaction
    and both are marked ``identifiable: False``.

    Returns
    -------
    components : dict[str | tuple, float]
    identifiability : dict[str | tuple | str, bool]
    raw_components : dict[str | tuple, float]
    """
    k = len(facet_list)
    all_axes = range(k)

    all_subsets: list[frozenset[int]] = []
    for r in range(1, k + 1):
        for combo in combinations(all_axes, r):
            all_subsets.append(frozenset(combo))

    effects = [s for s in all_subsets if s in selected] if selected is not None else list(all_subsets)

    grand_mean = float(means_array.mean())

    Q: dict[frozenset[int], float] = {}
    for S in all_subsets:
        axes_to_avg = tuple(i for i in all_axes if i not in S)
        marginal = means_array.mean(axis=axes_to_avg) if axes_to_avg else means_array
        n_per = n_reps * int(np.prod([n_levels[facet_list[i]] for i in all_axes if i not in S]))
        Q[S] = n_per * float(np.sum((marginal - grand_mean) ** 2))

    # Möbius inversion: SS_S = Σ_{T⊆S} (-1)^(|S|-|T|) Q_T
    SS: dict[frozenset[int], float] = {}
    for S in effects:
        ss = 0.0
        for r in range(1, len(S) + 1):
            for T_tuple in combinations(sorted(S), r):
                T = frozenset(T_tuple)
                ss += ((-1) ** (len(S) - r)) * Q[T]
        SS[S] = ss

    df_eff = {S: int(np.prod([n_levels[facet_list[i]] - 1 for i in S])) for S in effects}

    MS = {S: SS[S] / df_eff[S] if df_eff[S] > 0 else 0.0 for S in effects}

    def _ems_coeff(T: frozenset[int]) -> int:
        return n_reps * int(np.prod([n_levels[facet_list[i]] for i in all_axes if i not in T]))

    # Solve top-down; propagate raw (unclamped) estimates, clamp only at output.
    sorted_effects = sorted(effects, key=lambda s: (-len(s), sorted(s)))
    sigma2_raw: dict[frozenset[int], float] = {}

    for S in sorted_effects:
        rhs = MS[S] - ms_e
        for T in sorted_effects:
            if S < T and T in sigma2_raw:
                rhs -= _ems_coeff(T) * sigma2_raw[T]
        c = _ems_coeff(S)
        sigma2_raw[S] = rhs / c if c > 0 else 0.0

    full_interaction = frozenset(all_axes)
    components: dict[str | tuple, float] = {}
    raw_components: dict[str | tuple, float] = {}
    identifiability: dict[str | tuple, bool] = {}
    for S in effects:
        key = _effect_key(S, facet_list)
        raw_components[key] = sigma2_raw[S]
        components[key] = max(0.0, sigma2_raw[S])
        identifiability[key] = not (n_reps <= 1 and full_interaction == S)

    raw_components["residual"] = ms_e
    components["residual"] = max(0.0, ms_e)
    identifiability["residual"] = n_reps > 1

    return components, identifiability, raw_components


def _g_coefficients(
    vc_dict: dict,
    facet_sizes: dict[str, int],
    n_reps: int | None = None,
):
    """Compute universe-score variance, relative error, and absolute error.

    Parameters
    ----------
    vc_dict : dict
        Output of ``variance_components`` (multi-facet mode).
    facet_sizes : dict[str, int]
        Projected number of levels for each non-object facet.
    n_reps : int, optional
        Projected within-cell replications. Defaults to the observed
        ``vc_dict["n_reps"]``.

    Returns ``(sigma_p, rel_error, abs_error)`` as floats.
    """
    components = vc_dict["components"]
    object_facet = vc_dict["object_facet"]
    facets: list[str] = vc_dict["facets"]
    if n_reps is None:
        n_reps = vc_dict["n_reps"]

    non_object = [f for f in facets if f != object_facet]
    missing = set(non_object) - set(facet_sizes)
    if missing:
        raise ValueError(f"facet_sizes missing facets: {sorted(missing)}.")

    sigma_p = float(components[object_facet])
    rel_error = 0.0
    abs_error = 0.0

    for key, sigma2 in components.items():
        if key == object_facet:
            continue

        if key == "residual":
            divisor = n_reps
            for f in non_object:
                divisor *= facet_sizes[f]
            rel_error += sigma2 / divisor
            abs_error += sigma2 / divisor
            continue

        effect_facets = {key} if isinstance(key, str) else set(key)
        contains_object = object_facet in effect_facets
        non_obj_in = [f for f in effect_facets if f != object_facet]
        divisor = 1
        for f in non_obj_in:
            divisor *= facet_sizes[f]

        if contains_object:
            rel_error += sigma2 / divisor
            abs_error += sigma2 / divisor
        else:
            abs_error += sigma2 / divisor

    return sigma_p, rel_error, abs_error


def _bootstrap_resample(
    df: pd.DataFrame,
    object_col: str,
    vc_caller: Callable[[pd.DataFrame], dict],
    extract_components: Callable[[dict], dict],
    n_boot: int,
    ci: float,
    seed: int | None,
    point_estimate: dict | None = None,
) -> tuple[dict, dict, int, float]:
    """Bootstrap loop for both two-facet and multi-facet paths.

    Parameters
    ----------
    df : DataFrame
    object_col : str
        Column whose levels are resampled with replacement.
    vc_caller : callable(DataFrame) -> dict
    extract_components : callable(vc_dict) -> dict[key, float]
    n_boot, ci, seed : bootstrap parameters.
    point_estimate : dict, optional
        Pre-computed point estimate (avoids redundant computation).

    Returns
    -------
    samples_np : dict[key, ndarray]
    ci_dict : dict[key, tuple[float, float]]
    n_boot : int
    ci_level : float
    """
    import pandas as pd

    obj_ids = df[object_col].unique()
    n_obj = len(obj_ids)
    rng = np.random.default_rng(seed)
    by_obj = {oid: df[df[object_col] == oid] for oid in obj_ids}

    point = point_estimate if point_estimate is not None else vc_caller(df)
    component_keys = list(extract_components(point).keys())
    samples: dict = {k: [] for k in component_keys}

    for _ in range(n_boot):
        drawn = rng.choice(obj_ids, size=n_obj, replace=True)
        frames = []
        for j, oid in enumerate(drawn):
            block = by_obj[oid].copy()
            block[object_col] = f"__boot{j}__"
            frames.append(block)
        boot_df = pd.concat(frames, ignore_index=True)
        vc = vc_caller(boot_df)
        comps = extract_components(vc)
        for k in component_keys:
            samples[k].append(float(comps[k]))

    alpha = (1.0 - ci) / 2.0
    samples_np = {k: np.asarray(v, dtype=float) for k, v in samples.items()}
    ci_dict = {
        k: (
            float(np.quantile(samples_np[k], alpha)),
            float(np.quantile(samples_np[k], 1.0 - alpha)),
        )
        for k in component_keys
    }

    return samples_np, ci_dict, n_boot, float(ci)
