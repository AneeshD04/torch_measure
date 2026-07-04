# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Tests for multi-facet generalized G-theory extensions."""

import numpy as np
import pandas as pd
import pytest

from torch_measure.metrics.generalizability import (
    bootstrap_variance_components,
    d_study,
    g_coefficient,
    intraclass_correlation,
    variance_components,
)


def _synth_three_facet(
    n_p: int = 30,
    n_i: int = 10,
    n_r: int = 5,
    n_rep: int = 2,
    sigma_p: float = 1.0,
    sigma_i: float = 0.7,
    sigma_r: float = 0.4,
    sigma_pi: float = 0.5,
    sigma_pr: float = 0.3,
    sigma_ir: float = 0.2,
    sigma_pir: float = 0.15,
    sigma_e: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate balanced p x i x r x rep data with known variance components."""
    rng = np.random.default_rng(seed)
    a_p = rng.normal(0, sigma_p, n_p)
    a_i = rng.normal(0, sigma_i, n_i)
    a_r = rng.normal(0, sigma_r, n_r)
    a_pi = rng.normal(0, sigma_pi, (n_p, n_i))
    a_pr = rng.normal(0, sigma_pr, (n_p, n_r))
    a_ir = rng.normal(0, sigma_ir, (n_i, n_r))
    a_pir = rng.normal(0, sigma_pir, (n_p, n_i, n_r))
    e = rng.normal(0, sigma_e, (n_p, n_i, n_r, n_rep))

    y = (
        a_p[:, None, None, None]
        + a_i[None, :, None, None]
        + a_r[None, None, :, None]
        + a_pi[:, :, None, None]
        + a_pr[:, None, :, None]
        + a_ir[None, :, :, None]
        + a_pir[:, :, :, None]
        + e
    )

    rows = []
    for p in range(n_p):
        for i in range(n_i):
            for r in range(n_r):
                for rep in range(n_rep):
                    rows.append((f"s{p}", f"i{i}", f"r{r}", float(y[p, i, r, rep])))
    return pd.DataFrame(rows, columns=["subject", "item", "rater", "response"])


class TestMultiFacetVarianceComponents:
    def test_returns_expected_keys(self):
        df = _synth_three_facet(n_p=10, n_i=5, n_r=3, n_rep=2)
        vc = variance_components(
            df,
            response_col="response",
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        assert "components" in vc
        assert "object_facet" in vc
        assert "facets" in vc
        assert "n_levels" in vc
        assert "n_reps" in vc
        assert "identifiable" in vc
        assert "method" in vc
        assert vc["method"] == "moments"
        assert vc["object_facet"] == "subject"
        assert vc["facets"] == ["subject", "item", "rater"]
        assert vc["n_levels"] == {"subject": 10, "item": 5, "rater": 3}
        assert vc["n_reps"] == 2

    def test_all_components_present(self):
        df = _synth_three_facet(n_p=10, n_i=5, n_r=3, n_rep=2)
        vc = variance_components(
            df,
            response_col="response",
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        comps = vc["components"]
        expected_keys = {
            "subject",
            "item",
            "rater",
            ("subject", "item"),
            ("subject", "rater"),
            ("item", "rater"),
            ("subject", "item", "rater"),
            "residual",
        }
        assert set(comps.keys()) == expected_keys

    def test_three_facet_component_recovery(self):
        """Henderson Method I should recover realized sample variances."""
        rng = np.random.default_rng(42)
        n_p, n_i, n_r, n_rep = 80, 20, 8, 3

        a_p = rng.normal(0, 1.0, n_p)
        a_i = rng.normal(0, 0.7, n_i)
        a_r = rng.normal(0, 0.4, n_r)
        a_pi = rng.normal(0, 0.5, (n_p, n_i))
        a_pr = rng.normal(0, 0.3, (n_p, n_r))
        a_ir = rng.normal(0, 0.2, (n_i, n_r))
        a_pir = rng.normal(0, 0.15, (n_p, n_i, n_r))
        e = rng.normal(0, 0.3, (n_p, n_i, n_r, n_rep))

        y = (
            a_p[:, None, None, None]
            + a_i[None, :, None, None]
            + a_r[None, None, :, None]
            + a_pi[:, :, None, None]
            + a_pr[:, None, :, None]
            + a_ir[None, :, :, None]
            + a_pir[:, :, :, None]
            + e
        )

        rows = []
        for p in range(n_p):
            for i in range(n_i):
                for r in range(n_r):
                    for rep in range(n_rep):
                        rows.append((f"s{p}", f"i{i}", f"r{r}", float(y[p, i, r, rep])))
        df = pd.DataFrame(rows, columns=["subject", "item", "rater", "response"])

        vc = variance_components(
            df,
            response_col="response",
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        c = vc["components"]

        assert c["subject"] == pytest.approx(a_p.var(ddof=1), rel=0.10)
        assert c["item"] == pytest.approx(a_i.var(ddof=1), rel=0.25)
        assert c["rater"] == pytest.approx(a_r.var(ddof=1), rel=0.40)
        assert c[("subject", "item")] == pytest.approx(a_pi.var(ddof=1), rel=0.15)
        assert c[("subject", "rater")] == pytest.approx(a_pr.var(ddof=1), rel=0.20)
        assert c["residual"] == pytest.approx(e.var(ddof=1), rel=0.10)

    def test_components_non_negative(self):
        df = _synth_three_facet(n_p=10, n_i=5, n_r=3, n_rep=2)
        vc = variance_components(
            df,
            response_col="response",
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        for v in vc["components"].values():
            assert v >= 0.0

    def test_one_obs_per_cell_residual_unidentifiable(self):
        df = _synth_three_facet(n_p=15, n_i=5, n_r=3, n_rep=1)
        vc = variance_components(
            df,
            response_col="response",
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        assert vc["identifiable"]["residual"] is False
        assert vc["components"]["residual"] == 0.0
        assert vc["n_reps"] == 1

    def test_missing_cells_raises(self):
        df = _synth_three_facet(n_p=10, n_i=5, n_r=3, n_rep=2)
        mask = (df["subject"] == "s0") & (df["item"] == "i0") & (df["rater"] == "r0")
        df = df[~mask]
        with pytest.raises(ValueError, match="not fully crossed"):
            variance_components(
                df,
                response_col="response",
                facet_cols=["subject", "item", "rater"],
                object_facet="subject",
            )

    def test_unequal_replication_raises(self):
        df = _synth_three_facet(n_p=10, n_i=5, n_r=3, n_rep=2)
        extra = pd.DataFrame(
            [("s0", "i0", "r0", 1.0)],
            columns=["subject", "item", "rater", "response"],
        )
        df = pd.concat([df, extra], ignore_index=True)
        with pytest.raises(ValueError, match="Unequal replication"):
            variance_components(
                df,
                response_col="response",
                facet_cols=["subject", "item", "rater"],
                object_facet="subject",
            )

    def test_reml_rejected(self):
        df = _synth_three_facet(n_p=5, n_i=3, n_r=2, n_rep=2)
        with pytest.raises(NotImplementedError):
            variance_components(
                df,
                response_col="response",
                method="reml",
                facet_cols=["subject", "item", "rater"],
                object_facet="subject",
            )

    def test_object_facet_required(self):
        df = _synth_three_facet(n_p=5, n_i=3, n_r=2, n_rep=2)
        with pytest.raises(ValueError, match="object_facet is required"):
            variance_components(
                df,
                response_col="response",
                facet_cols=["subject", "item", "rater"],
            )

    def test_object_facet_must_be_in_facet_cols(self):
        df = _synth_three_facet(n_p=5, n_i=3, n_r=2, n_rep=2)
        with pytest.raises(ValueError, match="not in facet_cols"):
            variance_components(
                df,
                response_col="response",
                facet_cols=["subject", "item", "rater"],
                object_facet="bogus",
            )

    def test_too_few_facets_raises(self):
        df = _synth_three_facet(n_p=5, n_i=3, n_r=2, n_rep=2)
        with pytest.raises(ValueError, match="at least 2 facets"):
            variance_components(
                df,
                response_col="response",
                facet_cols=["subject"],
                object_facet="subject",
            )

    def test_selected_interactions(self):
        df = _synth_three_facet(n_p=15, n_i=5, n_r=3, n_rep=2)
        vc = variance_components(
            df,
            response_col="response",
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
            interactions=[("subject", "item"), ("subject", "rater")],
        )
        comps = vc["components"]
        # Main effects + selected interactions + residual
        assert "subject" in comps
        assert "item" in comps
        assert "rater" in comps
        assert ("subject", "item") in comps
        assert ("subject", "rater") in comps
        assert "residual" in comps
        # Unselected interactions should be absent
        assert ("item", "rater") not in comps
        assert ("subject", "item", "rater") not in comps


class TestMultiFacetGCoefficient:
    def _vc(self) -> dict:
        return {
            "components": {
                "subject": 1.0,
                "item": 0.5,
                "rater": 0.3,
                ("subject", "item"): 0.2,
                ("subject", "rater"): 0.1,
                ("item", "rater"): 0.15,
                ("subject", "item", "rater"): 0.05,
                "residual": 0.1,
            },
            "object_facet": "subject",
            "facets": ["subject", "item", "rater"],
            "n_levels": {"subject": 30, "item": 10, "rater": 5},
            "n_reps": 2,
            "identifiable": {
                "subject": True,
                "item": True,
                "rater": True,
                ("subject", "item"): True,
                ("subject", "rater"): True,
                ("item", "rater"): True,
                ("subject", "item", "rater"): True,
                "residual": True,
            },
            "method": "moments",
        }

    def test_in_unit_interval(self):
        vc = self._vc()
        for t in ("relative", "absolute"):
            g = g_coefficient(vc, facet_sizes={"item": 10, "rater": 3}, type=t)
            assert 0.0 <= g <= 1.0

    def test_relative_ge_absolute(self):
        vc = self._vc()
        ss = {"item": 10, "rater": 3}
        g_rel = g_coefficient(vc, facet_sizes=ss, type="relative")
        g_abs = g_coefficient(vc, facet_sizes=ss, type="absolute")
        assert g_rel >= g_abs

    def test_g_grows_with_sample_size(self):
        vc = self._vc()
        g_small = g_coefficient(vc, facet_sizes={"item": 5, "rater": 2}, type="absolute")
        g_large = g_coefficient(vc, facet_sizes={"item": 30, "rater": 10}, type="absolute")
        assert g_large > g_small

    def test_missing_facet_in_facet_sizes_raises(self):
        vc = self._vc()
        with pytest.raises(ValueError, match="facet_sizes missing"):
            g_coefficient(vc, facet_sizes={"item": 10}, type="absolute")

    def test_invalid_sample_size_raises(self):
        vc = self._vc()
        with pytest.raises(ValueError, match="facet_sizes must be >= 1"):
            g_coefficient(vc, facet_sizes={"item": 0, "rater": 3}, type="absolute")

    def test_manual_two_facet_computation(self):
        """Verify generalized G matches manual formula for a two-facet case."""
        vc = {
            "components": {
                "subject": 1.0,
                "item": 0.5,
                ("subject", "item"): 0.3,
                "residual": 0.2,
            },
            "object_facet": "subject",
            "facets": ["subject", "item"],
            "n_levels": {"subject": 20, "item": 10},
            "n_reps": 2,
            "identifiable": {k: True for k in ["subject", "item", ("subject", "item"), "residual"]},
            "method": "moments",
        }
        n_i = 15
        rel_err = 0.3 / n_i + 0.2 / (n_i * 2)
        abs_err = 0.5 / n_i + rel_err

        g_rel = g_coefficient(vc, facet_sizes={"item": n_i}, type="relative")
        g_abs = g_coefficient(vc, facet_sizes={"item": n_i}, type="absolute")

        assert g_rel == pytest.approx(1.0 / (1.0 + rel_err))
        assert g_abs == pytest.approx(1.0 / (1.0 + abs_err))


class TestMultiFacetDStudy:
    def _vc(self) -> dict:
        return {
            "components": {
                "subject": 1.0,
                "item": 0.5,
                "rater": 0.3,
                ("subject", "item"): 0.2,
                ("subject", "rater"): 0.1,
                ("item", "rater"): 0.15,
                ("subject", "item", "rater"): 0.05,
                "residual": 0.1,
            },
            "object_facet": "subject",
            "facets": ["subject", "item", "rater"],
            "n_levels": {"subject": 30, "item": 10, "rater": 5},
            "n_reps": 2,
            "identifiable": {
                k: True
                for k in [
                    "subject",
                    "item",
                    "rater",
                    ("subject", "item"),
                    ("subject", "rater"),
                    ("item", "rater"),
                    ("subject", "item", "rater"),
                    "residual",
                ]
            },
            "method": "moments",
        }

    def test_shape_and_columns(self):
        vc = self._vc()
        result = d_study(vc, design_grid={"item": [10, 20], "rater": [1, 2, 3]})
        assert len(result) == 2 * 3
        assert "n_item" in result.columns
        assert "n_rater" in result.columns
        assert "g_relative" in result.columns
        assert "g_absolute" in result.columns
        assert "se_relative" in result.columns
        assert "se_absolute" in result.columns

    def test_g_increases_with_facet_sizes(self):
        vc = self._vc()
        result = d_study(vc, design_grid={"item": [5, 10, 20, 40], "rater": [3]})
        g_vals = result.sort_values("n_item")["g_absolute"].to_numpy()
        assert np.all(np.diff(g_vals) > 0)

    def test_se_decreases_with_facet_sizes(self):
        vc = self._vc()
        result = d_study(vc, design_grid={"item": [5, 10, 20, 40], "rater": [3]})
        se_vals = result.sort_values("n_item")["se_absolute"].to_numpy()
        assert np.all(np.diff(se_vals) < 0)

    def test_empty_grid_raises(self):
        vc = self._vc()
        with pytest.raises(ValueError, match="non-empty"):
            d_study(vc, design_grid={"item": [], "rater": [1]})


# bootstrap (generalized)
class TestMultiFacetBootstrap:
    def _df(self, seed: int = 0) -> pd.DataFrame:
        return _synth_three_facet(n_p=15, n_i=5, n_r=3, n_rep=2, seed=seed)

    def test_output_structure(self):
        out = bootstrap_variance_components(
            self._df(),
            response_col="response",
            n_boot=20,
            seed=42,
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        assert "components" in out
        assert "ci" in out
        assert "samples" in out
        assert out["n_boot"] == 20
        assert out["ci_level"] == 0.95

        for k in out["components"]:
            assert k in out["ci"]
            lo, hi = out["ci"][k]
            assert lo <= hi
            assert k in out["samples"]

    def test_point_matches_variance_components(self):
        df = self._df()
        out = bootstrap_variance_components(
            df,
            response_col="response",
            n_boot=5,
            seed=0,
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        vc = variance_components(
            df,
            response_col="response",
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        for k in vc["components"]:
            assert out["components"][k] == pytest.approx(vc["components"][k])

    def test_reproducibility(self):
        df = self._df()
        kwargs = dict(
            response_col="response",
            n_boot=15,
            seed=123,
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        a = bootstrap_variance_components(df, **kwargs)
        b = bootstrap_variance_components(df, **kwargs)
        for k in a["components"]:
            np.testing.assert_allclose(a["samples"][k], b["samples"][k])

    def test_different_seeds_differ(self):
        df = self._df()
        common = dict(
            response_col="response",
            n_boot=20,
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        a = bootstrap_variance_components(df, seed=1, **common)
        b = bootstrap_variance_components(df, seed=2, **common)
        any_diff = any(not np.allclose(a["samples"][k], b["samples"][k]) for k in a["components"])
        assert any_diff

    def test_ci_brackets_point_estimate_for_dominant_component(self):
        df = self._df()
        out = bootstrap_variance_components(
            df,
            response_col="response",
            n_boot=200,
            seed=42,
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        # The object facet ("subject") should be the dominant component
        k = "subject"
        lo, hi = out["ci"][k]
        assert lo <= out["components"][k] <= hi

    def test_custom_ci_level(self):
        df = self._df()
        out = bootstrap_variance_components(
            df,
            response_col="response",
            n_boot=20,
            ci=0.90,
            seed=0,
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        assert out["ci_level"] == 0.90

    def test_invalid_n_boot_raises(self):
        df = self._df()
        with pytest.raises(ValueError, match="n_boot"):
            bootstrap_variance_components(
                df,
                response_col="response",
                n_boot=0,
                facet_cols=["subject", "item", "rater"],
                object_facet="subject",
            )

    def test_invalid_ci_raises(self):
        df = self._df()
        with pytest.raises(ValueError, match="ci"):
            bootstrap_variance_components(
                df,
                response_col="response",
                n_boot=10,
                ci=1.5,
                facet_cols=["subject", "item", "rater"],
                object_facet="subject",
            )

    def test_g_coefficient_ci_via_samples(self):
        df = self._df()
        out = bootstrap_variance_components(
            df,
            response_col="response",
            n_boot=50,
            seed=7,
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        # Derive G-coefficient from each bootstrap draw
        g_boots = []
        for idx in range(out["n_boot"]):
            boot_vc = {
                "components": {k: out["samples"][k][idx] for k in out["components"]},
                "object_facet": out["object_facet"],
                "facets": out["facets"],
                "n_levels": out["n_levels"],
                "n_reps": out["n_reps"],
            }
            g_boots.append(g_coefficient(boot_vc, facet_sizes={"item": 5, "rater": 3}, type="absolute"))
        assert all(0.0 <= g <= 1.0 for g in g_boots)


class TestICCGeneralizedRejection:
    def test_rejects_generalized_components(self):
        df = _synth_three_facet(n_p=10, n_i=5, n_r=3, n_rep=2)
        vc = variance_components(
            df,
            response_col="response",
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        with pytest.raises(TypeError, match="does not support multi-facet"):
            intraclass_correlation(vc, form="ICC3k")


class TestMultiFacetEndToEnd:
    def test_pipeline(self):
        """variance_components -> g_coefficient -> d_study for 3 facets."""
        df = _synth_three_facet(n_p=20, n_i=8, n_r=4, n_rep=2, seed=99)
        vc = variance_components(
            df,
            response_col="response",
            facet_cols=["subject", "item", "rater"],
            object_facet="subject",
        )
        g = g_coefficient(vc, facet_sizes={"item": 8, "rater": 4}, type="absolute")
        assert 0.0 < g < 1.0

        proj = d_study(vc, design_grid={"item": [8, 16], "rater": [2, 4]})
        assert len(proj) == 4
        assert all(0 < v <= 1 for v in proj["g_absolute"])
