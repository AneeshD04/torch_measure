# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Generalizability-theory reliability for two-way crossed designs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def variance_components(
    response_matrix: pd.DataFrame,
    subject_col: str = "subject_id",
    item_col: str = "item_id",
    trial_col: str = "trial",
    response_col: str = "response",
    method: str = "moments",
) -> dict:
    """Decompose Var(response) into subject, item, subject x item, and residual facets.

    Henderson Method I (moments-based ANOVA estimator) on a person x item x
    replication crossed design. Negative variance estimates are clamped to 0.
    With one observation per cell, residual is unidentifiable.

    Parameters
    ----------
    response_matrix : pandas.DataFrame
        Long-form responses with columns ``subject_col``, ``item_col``,
        ``trial_col``, ``response_col``.
    subject_col, item_col, trial_col, response_col : str
        Column names; defaults match the measurement-db long-form schema.
    method : {"moments"}
        Only ``"moments"`` is implemented in v1.

    Returns
    -------
    dict
        Keys: ``subject``, ``item``, ``subject_item``, ``residual`` (variances,
        floats), ``n_subjects``, ``n_items`` (ints), ``n_reps_harmonic``
        (float; harmonic mean of cell counts), ``identifiable`` (dict[str,
        bool]), ``method`` (str).
    """
    import pandas as pd

    if method == "reml":
        raise NotImplementedError("method='reml' not implemented in v1.")
    if method != "moments":
        raise ValueError(f"Unknown method: {method!r}.")

    required = {subject_col, item_col, trial_col, response_col}
    missing = required - set(response_matrix.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}.")

    df = response_matrix[[subject_col, item_col, trial_col, response_col]].dropna(subset=[response_col])
    if not pd.api.types.is_numeric_dtype(df[response_col]):
        raise ValueError(f"{response_col!r} column must be numeric.")

    n_p = int(df[subject_col].nunique())
    n_i = int(df[item_col].nunique())
    if n_p < 2 or n_i < 2:
        raise ValueError(f"Need at least 2 subjects and 2 items; got n_subjects={n_p}, n_items={n_i}.")

    cell = df.groupby([subject_col, item_col])[response_col].agg(["mean", "count"]).reset_index()
    cell = cell.rename(columns={"mean": "_cell_mean", "count": "_cell_count"})

    if len(cell) < n_p * n_i:
        raise ValueError(
            f"Unbalanced design: {len(cell)}/{n_p * n_i} cells observed. "
            f"Every (subject, item) cell must have at least one observation."
        )

    counts = cell["_cell_count"].to_numpy(dtype=float)
    n_r = float(len(counts) / np.sum(1.0 / counts))  # harmonic mean of cell counts
    has_replications = bool(np.any(counts > 1))

    cell_table = cell.pivot(index=subject_col, columns=item_col, values="_cell_mean").to_numpy(dtype=float)
    grand_mean = cell_table.mean()
    subj_mean = cell_table.mean(axis=1)
    item_mean = cell_table.mean(axis=0)

    # ANOVA on cell means; multiply by n_r to lift to observation-level SS.
    ss_p = n_r * n_i * float(np.sum((subj_mean - grand_mean) ** 2))
    ss_i = n_r * n_p * float(np.sum((item_mean - grand_mean) ** 2))
    ss_pi = n_r * float(np.sum((cell_table - grand_mean) ** 2)) - ss_p - ss_i

    if has_replications:
        merged = df.merge(cell[[subject_col, item_col, "_cell_mean"]], on=[subject_col, item_col])
        ss_e = float(((merged[response_col] - merged["_cell_mean"]) ** 2).sum())
        df_e = int(np.sum(counts - 1))
        ms_e = ss_e / df_e if df_e > 0 else 0.0
    else:
        ms_e = 0.0

    ms_p = ss_p / (n_p - 1)
    ms_i = ss_i / (n_i - 1)
    ms_pi = ss_pi / ((n_p - 1) * (n_i - 1))

    sigma2_e = max(0.0, ms_e)
    sigma2_pi = max(0.0, (ms_pi - ms_e) / n_r)
    sigma2_i = max(0.0, (ms_i - ms_pi) / (n_p * n_r))
    sigma2_p = max(0.0, (ms_p - ms_pi) / (n_i * n_r))

    return {
        "subject": sigma2_p,
        "item": sigma2_i,
        "subject_item": sigma2_pi,
        "residual": sigma2_e,
        "n_subjects": n_p,
        "n_items": n_i,
        "n_reps_harmonic": n_r,
        "identifiable": {
            "subject": True,
            "item": True,
            "subject_item": True,
            "residual": has_replications,
        },
        "method": "moments",
    }


def g_coefficient(
    variance_components: dict,
    n_items: int,
    n_reps: int = 1,
    type: str = "absolute",
) -> float:
    """Brennan (2001) G-coefficient under a person x item x replication design.

    Relative G uses ranking-only error (subject x item + residual); absolute G
    (Phi) also includes the item main effect.

    Parameters
    ----------
    variance_components : dict
        Output of :func:`variance_components`, or any dict with keys
        ``subject``, ``item``, ``subject_item``, ``residual``.
    n_items : int
        Number of items in the projected design (>= 1).
    n_reps : int
        Replications per cell in the projected design (>= 1).
    type : {"relative", "absolute"}
        Which G-coefficient to compute.

    Returns
    -------
    float
        G-coefficient in [0, 1]. 0.0 if the denominator is numerically zero.
    """
    required = {"subject", "item", "subject_item", "residual"}
    missing = required - set(variance_components)
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}.")
    if type not in {"relative", "absolute"}:
        raise ValueError(f"type must be 'relative' or 'absolute'; got {type!r}.")
    if n_items < 1 or n_reps < 1:
        raise ValueError(f"n_items and n_reps must be >= 1; got n_items={n_items}, n_reps={n_reps}.")

    s_p = float(variance_components["subject"])
    s_i = float(variance_components["item"])
    s_pi = float(variance_components["subject_item"])
    s_e = float(variance_components["residual"])

    err_relative = s_pi / n_items + s_e / (n_items * n_reps)
    err = (s_i / n_items + err_relative) if type == "absolute" else err_relative

    denom = s_p + err
    if denom < 1e-12:
        return 0.0
    return s_p / denom


def intraclass_correlation(
    variance_components: dict,
    form: str = "ICC3k",
    n_items: int | None = None,
) -> float:
    """Intraclass correlation coefficient from two-way variance components.

    Subjects are targets, items are raters. ICC2/ICC3 are single-rater
    (absolute agreement / consistency); ICC2k/ICC3k average over k raters and
    equal the absolute / relative :func:`g_coefficient` at ``n_reps=1``.
    One-way forms (ICC1) need a one-way model and are not supported here.

    Parameters
    ----------
    variance_components : dict
        Output of :func:`variance_components`, or any dict with keys
        ``subject``, ``item``, ``subject_item``, ``residual``.
    form : {"ICC2", "ICC3", "ICC2k", "ICC3k"}
        Which coefficient to compute. The ``k`` forms average over raters.
    n_items : int | None
        Number of raters k for the ``k`` forms; defaults to
        ``variance_components["n_items"]``.

    Returns
    -------
    float
        ICC in [0, 1]. 0.0 if the denominator is numerically zero.
    """
    required = {"subject", "item", "subject_item", "residual"}
    missing = required - set(variance_components)
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}.")
    if form in {"ICC1", "ICC1k"}:
        raise ValueError(f"{form} requires a one-way model and is not supported; use ICC2/ICC3/ICC2k/ICC3k.")
    if form not in {"ICC2", "ICC3", "ICC2k", "ICC3k"}:
        raise ValueError(f"Unknown form: {form!r}. Expected one of ICC2, ICC3, ICC2k, ICC3k.")

    s_p = float(variance_components["subject"])
    s_i = float(variance_components["item"])
    s_pi = float(variance_components["subject_item"])
    s_e = float(variance_components["residual"])

    averaged = form.endswith("k")
    absolute = form in {"ICC2", "ICC2k"}

    if averaged:
        k = n_items if n_items is not None else int(variance_components["n_items"])
        if k < 1:
            raise ValueError(f"n_items must be >= 1; got {k}.")
    else:
        k = 1

    err = (s_i + s_pi + s_e) if absolute else (s_pi + s_e)
    err = err / k

    denom = s_p + err
    if denom < 1e-12:
        return 0.0
    return s_p / denom


def d_study(
    variance_components: dict,
    n_items_grid: Sequence[int],
    n_reps_grid: Sequence[int],
) -> pd.DataFrame:
    """Project G-coefficients and SEs over a (n_items, n_reps) design grid.

    Parameters
    ----------
    variance_components : dict
        Output of :func:`variance_components`.
    n_items_grid, n_reps_grid : sequence of int
        Candidate design dimensions to project.

    Returns
    -------
    pandas.DataFrame
        One row per (n_items, n_reps) cell with columns ``n_items``,
        ``n_reps``, ``g_relative``, ``g_absolute``, ``se_relative``,
        ``se_absolute``.
    """
    import pandas as pd

    if len(n_items_grid) == 0 or len(n_reps_grid) == 0:
        raise ValueError("n_items_grid and n_reps_grid must be non-empty.")

    s_i = float(variance_components["item"])
    s_pi = float(variance_components["subject_item"])
    s_e = float(variance_components["residual"])

    rows = []
    for n_items in n_items_grid:
        for n_reps in n_reps_grid:
            err_relative = s_pi / n_items + s_e / (n_items * n_reps)
            err_absolute = s_i / n_items + err_relative
            rows.append(
                {
                    "n_items": int(n_items),
                    "n_reps": int(n_reps),
                    "g_relative": g_coefficient(variance_components, n_items, n_reps, "relative"),
                    "g_absolute": g_coefficient(variance_components, n_items, n_reps, "absolute"),
                    "se_relative": float(np.sqrt(err_relative)),
                    "se_absolute": float(np.sqrt(err_absolute)),
                }
            )
    return pd.DataFrame(rows)


def bayesian_variance_components(
    response_matrix: pd.DataFrame,
    subject_col: str = "subject_id",
    item_col: str = "item_id",
    trial_col: str = "trial",
    response_col: str = "response",
    n_warmup: int = 200,
    n_samples: int = 500,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Full-Bayes variance components via Hamiltonian Monte Carlo (textbook Ch 5).

    Fits the hierarchical Bernoulli model
        logit P(Y_ij=1) = mu + theta_i - beta_j + gamma_ij
    where theta_i ~ N(0, sigma_p), beta_j ~ N(0, sigma_i), gamma_ij ~ N(0, sigma_pi),
    and sigma_p, sigma_i, sigma_pi ~ HalfNormal(1) using NUTS (No-U-Turn Sampler).

    Returns the same top-level keys as variance_components() so all downstream
    functions (g_coefficient, d_study, intraclass_correlation) work unchanged.
    Unlike variance_components(), residual is always 0.0 and unidentifiable:
    the Bernoulli likelihood has no separate additive noise term, so there is
    nothing analogous to observation-level residual variance to estimate,
    regardless of how many replications per cell are present.

    Parameters
    ----------
    response_matrix : pandas.DataFrame
        Long-form responses with columns subject_col, item_col, trial_col, response_col.
    subject_col, item_col, trial_col, response_col : str
        Column names; defaults match measurement-db long-form schema.
    n_warmup : int
        NUTS warmup (adaptation) steps. 200 is the minimum; 500+ is safer for
        large datasets.
    n_samples : int
        Post-warmup samples to draw. 500 gives stable posterior means; 1000+ for
        accurate credible intervals.
    seed : int | None
        Pyro RNG seed for reproducibility.
    verbose : bool
        Show NUTS progress bar.

    Returns
    -------
    dict
        Top-level keys match variance_components() for drop-in compatibility:
        subject, item, subject_item, residual, n_subjects, n_items,
        n_reps_harmonic, identifiable, method.
        Additional keys: posterior_samples, credible_intervals, diagnostics,
        n_warmup, n_samples, ci_level.
    """
    import pandas as pd
    import pyro
    import pyro.distributions as dist
    import torch
    from pyro.infer import MCMC, NUTS

    # Non-centered hierarchical Bayesian G-theory model (textbook Ch 5, logit link).
    # Non-centered form: theta_i = sigma_p * z_p[i] instead of theta_i ~ N(0, sigma_p).
    # Required for NUTS to avoid funnel geometry when sigma_pi is near zero.
    def _gtheory_model(s_idx, i_idx, n_s, n_i, obs=None):
        mu = pyro.sample("mu", dist.Normal(0.0, 2.0))
        sigma_p = pyro.sample("sigma_p", dist.HalfNormal(1.0))
        sigma_i = pyro.sample("sigma_i", dist.HalfNormal(1.0))
        sigma_pi = pyro.sample("sigma_pi", dist.HalfNormal(1.0))
        z_p = pyro.sample("z_p", dist.Normal(0.0, 1.0).expand([n_s]).to_event(1))
        z_i = pyro.sample("z_i", dist.Normal(0.0, 1.0).expand([n_i]).to_event(1))
        z_pi = pyro.sample("z_pi", dist.Normal(0.0, 1.0).expand([n_s, n_i]).to_event(2))
        # Textbook formula: logit P(Y_ij=1) = mu + theta_i - beta_j + gamma_ij
        logit = mu + sigma_p * z_p[s_idx] - sigma_i * z_i[i_idx] + sigma_pi * z_pi[s_idx, i_idx]
        with pyro.plate("obs", s_idx.shape[0]):
            pyro.sample("response", dist.Bernoulli(logits=logit), obs=obs)

    # -- Input validation (mirrors variance_components()) --------------------
    required = {subject_col, item_col, trial_col, response_col}
    missing = required - set(response_matrix.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}.")

    df = response_matrix[[subject_col, item_col, trial_col, response_col]].dropna(subset=[response_col])
    if not pd.api.types.is_numeric_dtype(df[response_col]):
        raise ValueError(f"{response_col!r} column must be numeric.")

    n_p = int(df[subject_col].nunique())
    n_i = int(df[item_col].nunique())
    if n_p < 2 or n_i < 2:
        raise ValueError(f"Need at least 2 subjects and 2 items; got n_subjects={n_p}, n_items={n_i}.")

    n_obs_per_cell = df.groupby([subject_col, item_col])[response_col].count()
    if len(n_obs_per_cell) < n_p * n_i:
        raise ValueError(
            f"Unbalanced design: {len(n_obs_per_cell)}/{n_p * n_i} cells observed. "
            f"Every (subject, item) cell must have at least one observation."
        )

    counts = n_obs_per_cell.to_numpy(dtype=float)
    n_r = float(len(counts) / np.sum(1.0 / counts))  # harmonic mean of cell counts

    # -- Index encoding -------------------------------------------------------
    subject_ids = sorted(df[subject_col].unique())
    item_ids = sorted(df[item_col].unique())
    s_to_idx = {s: k for k, s in enumerate(subject_ids)}
    i_to_idx = {it: k for k, it in enumerate(item_ids)}

    subject_idx = torch.tensor([s_to_idx[s] for s in df[subject_col]], dtype=torch.long)
    item_idx = torch.tensor([i_to_idx[it] for it in df[item_col]], dtype=torch.long)
    response = torch.tensor(df[response_col].to_numpy(dtype="float32"), dtype=torch.float32)

    # -- NUTS sampling -------------------------------------------------------
    if seed is not None:
        pyro.set_rng_seed(seed)

    kernel = NUTS(_gtheory_model, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_samples=n_samples,
        warmup_steps=n_warmup,
        disable_progbar=not verbose,
    )
    mcmc.run(subject_idx, item_idx, n_p, n_i, obs=response)

    # -- Extract posterior samples -------------------------------------------
    raw = mcmc.get_samples()
    sigma_p_samp = raw["sigma_p"].numpy()
    sigma_i_samp = raw["sigma_i"].numpy()
    sigma_pi_samp = raw["sigma_pi"].numpy()

    sigma2_p_samp = sigma_p_samp**2
    sigma2_i_samp = sigma_i_samp**2
    sigma2_pi_samp = sigma_pi_samp**2

    alpha = 0.025
    ci_p = (float(np.quantile(sigma2_p_samp, alpha)), float(np.quantile(sigma2_p_samp, 1 - alpha)))
    ci_i = (float(np.quantile(sigma2_i_samp, alpha)), float(np.quantile(sigma2_i_samp, 1 - alpha)))
    ci_pi = (float(np.quantile(sigma2_pi_samp, alpha)), float(np.quantile(sigma2_pi_samp, 1 - alpha)))

    # -- Diagnostics ---------------------------------------------------------
    diag = mcmc.diagnostics()

    def _stat(site: str, key: str) -> float:
        if site not in diag:
            return float("nan")
        v = diag[site][key]
        return float(torch.as_tensor(v).float().mean().item())

    # diag["divergences"] maps each chain to the list of sample indices where
    # NUTS reported a divergent transition. (Pyro's MCMC has no
    # get_extra_fields() -- that is a NumPyro-only API.)
    n_divergences = sum(len(steps) for steps in diag.get("divergences", {}).values())

    return {
        # Drop-in compatible keys (same as variance_components() output)
        "subject": float(sigma2_p_samp.mean()),
        "item": float(sigma2_i_samp.mean()),
        "subject_item": float(sigma2_pi_samp.mean()),
        "residual": 0.0,
        "n_subjects": n_p,
        "n_items": n_i,
        "n_reps_harmonic": n_r,
        "identifiable": {
            "subject": True,
            "item": True,
            "subject_item": True,
            "residual": False,
        },
        "method": "hmc",
        # Bayesian-specific keys
        "posterior_samples": {
            "sigma_p": sigma_p_samp,
            "sigma_i": sigma_i_samp,
            "sigma_pi": sigma_pi_samp,
            "sigma2_p": sigma2_p_samp,
            "sigma2_i": sigma2_i_samp,
            "sigma2_pi": sigma2_pi_samp,
        },
        "credible_intervals": {
            "subject": ci_p,
            "item": ci_i,
            "subject_item": ci_pi,
        },
        "diagnostics": {
            "r_hat": {
                "sigma_p": _stat("sigma_p", "r_hat"),
                "sigma_i": _stat("sigma_i", "r_hat"),
                "sigma_pi": _stat("sigma_pi", "r_hat"),
            },
            "n_eff": {
                "sigma_p": _stat("sigma_p", "n_eff"),
                "sigma_i": _stat("sigma_i", "n_eff"),
                "sigma_pi": _stat("sigma_pi", "n_eff"),
            },
            "divergences": n_divergences,
        },
        "n_warmup": n_warmup,
        "n_samples": n_samples,
        "ci_level": 0.95,
    }


def bootstrap_variance_components(
    response_matrix: pd.DataFrame,
    subject_col: str = "subject_id",
    item_col: str = "item_id",
    trial_col: str = "trial",
    response_col: str = "response",
    method: str = "moments",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict:
    """Nonparametric bootstrap CIs for variance components by resampling subjects.

    Subjects are the exchangeable unit: each bootstrap draw samples
    ``n_subjects`` subjects with replacement, relabels duplicates so each
    draw is treated as a distinct unit, and re-fits :func:`variance_components`.
    Percentile CIs are reported. The full bootstrap distribution for each
    component is also returned so callers can derive CIs for any function of
    the components (e.g. :func:`g_coefficient`, :func:`intraclass_correlation`).

    Parameters
    ----------
    response_matrix : pandas.DataFrame
        Long-form responses, same schema as :func:`variance_components`.
    subject_col, item_col, trial_col, response_col, method : str
        Forwarded to :func:`variance_components`.
    n_boot : int
        Number of bootstrap replicates (>= 1).
    ci : float
        Confidence level in (0, 1).
    seed : int | None
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    dict
        Keys: ``subject``, ``item``, ``subject_item``, ``residual`` (point
        estimates on the original sample), ``n_subjects``, ``n_items``,
        ``n_reps_harmonic``, ``identifiable``, ``method`` (as in
        :func:`variance_components`), plus ``ci`` (dict[str, tuple[float,
        float]] per component), ``samples`` (dict[str, numpy.ndarray] of
        length ``n_boot``), ``n_boot`` (int), and ``ci_level`` (float).
    """
    import pandas as pd

    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}.")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1); got {ci}.")

    point = variance_components(
        response_matrix,
        subject_col=subject_col,
        item_col=item_col,
        trial_col=trial_col,
        response_col=response_col,
        method=method,
    )

    subject_ids = response_matrix[subject_col].unique()
    n_subjects = len(subject_ids)
    rng = np.random.default_rng(seed)

    by_subject = {sid: response_matrix[response_matrix[subject_col] == sid] for sid in subject_ids}

    component_keys = ("subject", "item", "subject_item", "residual")
    samples: dict[str, list[float]] = {k: [] for k in component_keys}

    for _ in range(n_boot):
        drawn = rng.choice(subject_ids, size=n_subjects, replace=True)
        frames = []
        for j, sid in enumerate(drawn):
            block = by_subject[sid].copy()
            block[subject_col] = f"__boot{j}__"
            frames.append(block)
        boot_df = pd.concat(frames, ignore_index=True)
        try:
            vc = variance_components(
                boot_df,
                subject_col=subject_col,
                item_col=item_col,
                trial_col=trial_col,
                response_col=response_col,
                method=method,
            )
        except ValueError:
            continue  # skip degenerate draws (e.g. accidentally singular cell counts)
        for k in component_keys:
            samples[k].append(float(vc[k]))

    alpha = (1.0 - ci) / 2.0
    samples_np = {k: np.asarray(v, dtype=float) for k, v in samples.items()}
    ci_dict = {
        k: (
            float(np.quantile(samples_np[k], alpha)),
            float(np.quantile(samples_np[k], 1.0 - alpha)),
        )
        if len(samples_np[k]) > 0
        else (float("nan"), float("nan"))
        for k in component_keys
    }

    return {
        **point,
        "ci": ci_dict,
        "samples": samples_np,
        "n_boot": int(n_boot),
        "ci_level": float(ci),
    }
