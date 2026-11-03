import copy
import inspect

import numpy as np
import pandas as pd
import pytest

import pyfylr as pf

COLUMNS = [
    "model_id", "role", "wave", "model", "K", "covariance",
    "measurement_invariant", "means_invariant", "covariance_invariant", "transitions",
    "loglik", "n_parameters", "bic", "sabic", "aic", "ic_supported", "ic_reason",
    "n_best", "n_starts", "n_converged_starts", "converged", "status",
    "min_expected_n", "min_expected_proportion", "small_state_flag", "small_state_rule",
    "min_expected_transitions", "transition_note",
    "entropy_wave", "entropy_sequence", "mean_max_posterior", "assignment_note",
    "blrt", "blrt_reason", "lmr", "lmr_reason",
    "n_boundary", "regularized", "min_eigenvalue", "boundary_warning", "regularization_warning",
    "reliable", "unreliable_reason", "inference_warning", "criterion_note", "selected", "selection_rationale",
    "comparable", "incomparable",
]


def gaussian_sample():
    rng = np.random.default_rng(1)
    y1 = rng.normal(size=100)
    y2 = 0.6 * y1 + rng.normal(0.2, size=100)
    return pd.DataFrame({"y1": y1, "y2": y2})


def one_state_grid():
    return pf.fit_grid(
        gaussian_sample(), ["y1", "y2"], n_states=[1], covariance=["EEI", "VVV"],
        n_starts=2, max_iter=80, tol=1e-8, screen_iter=0, seed=1,
    )


def test_compare_models_stays_a_report_and_select_model_applies_the_rule():
    for fn in (pf.fit_lpa, pf.fit_lca, pf.fit_lta, pf.fit_grid, pf.compare_models, pf.enumerate_waves):
        names = inspect.signature(fn).parameters
        assert "select_best" not in names
        assert "auto" not in names
        assert not any("recommended" in name for name in names)
    assert "auto" in inspect.signature(pf.select_model).parameters
    grid = one_state_grid()
    tab = grid.comparison
    assert list(tab.columns) == COLUMNS
    assert not tab["selected"].any()
    assert set(tab["selection_rationale"]) == {""}
    assert not any(name in tab.columns for name in ("recommended_k", "select_best", "best_k"))
    assert not hasattr(grid, "recommended_k")
    sel = pf.select_model(grid)
    assert not sel.comparison["selected"].any()
    assert list(sel.comparison.columns) == COLUMNS
    chosen = sel.candidates[sel.candidates["selected"]]
    assert not (~sel.candidates["hard_pass"] & sel.candidates["selected"]).any()
    assert not (sel.candidates["selected"] & (sel.candidates["n_best"] < 2)).any()
    if len(chosen):
        assert float(chosen["second_seed_abs_gap"].iloc[0]) <= float(chosen["replicate_tol"].iloc[0])
        assert chosen["rejection_reason"].iloc[0] == ""
    assert not sel.disagreement["selects"].any()
    assert not any("n_criteria" in name or "favoring" in name for name in sel.disagreement.columns)
    assert not any("n_criteria" in name or "favoring" in name for name in sel.comparison.columns)


def test_no_vote_count_across_indices():
    tab = one_state_grid().comparison
    assert not any("n_criteria" in name or "favoring" in name for name in tab.columns)
    assert not tab["criterion_note"].str.contains(r"criteria favoring|\d+ criteria").any()
    assert tab["criterion_note"].nunique() == 1


def test_small_class_cutoff_is_opt_in():
    grid = one_state_grid()
    tab = grid.comparison
    assert np.isfinite(tab["min_expected_n"]).all()
    assert np.isfinite(tab["min_expected_proportion"]).all()
    assert not tab["small_state_flag"].any()
    assert set(tab["small_state_rule"]) == {""}
    assert not tab["unreliable_reason"].str.contains("small_class").any()
    flagged = pf.compare_models(grid.fits, min_proportion=2)
    assert len(flagged) == len(tab)
    assert flagged["small_state_flag"].all()
    assert "min_proportion=2" in flagged["small_state_rule"].iloc[0]


def test_changing_entropy_leaves_eligibility_order_and_other_columns_unchanged():
    grid = one_state_grid()
    base = grid.comparison.copy()
    assert base["model_id"].tolist() == [1, 2]
    assert base["covariance"].tolist() == ["EEI", "VVV"]
    assert "not used to choose" in base["assignment_note"].iloc[0]
    clones = []
    for fit, wave, sequence in zip(grid.fits, (0.01, 0.99), (0.02, 0.98)):
        clone = copy.deepcopy(fit)
        clone.result["entropy_wave"] = wave
        clone.result["entropy_sequence"] = sequence
        clones.append(clone)
    again = pf.compare_models(clones)
    assert again["entropy_wave"].tolist() == [0.01, 0.99]
    assert again["model_id"].tolist() == base["model_id"].tolist()
    kept = [name for name in base.columns if name not in ("entropy_wave", "entropy_sequence")]
    pd.testing.assert_frame_equal(again[kept].reset_index(drop=True), base[kept].reset_index(drop=True))
    flipped = pf.compare_models([clones[1], clones[0]])
    assert flipped["entropy_wave"].tolist() == [0.99, 0.01]
    assert flipped["model_id"].tolist() == [1, 2]
    assert flipped["reliable"].tolist() == list(reversed(again["reliable"].tolist()))
    assert flipped["unreliable_reason"].tolist() == list(reversed(again["unreliable_reason"].tolist()))
    assert flipped["inference_warning"].tolist() == list(reversed(again["inference_warning"].tolist()))
    injected = base.copy()
    injected["entropy_wave"] = [0.01, 0.99]
    injected["entropy_sequence"] = [0.02, 0.98]
    pd.testing.assert_frame_equal(injected[kept].reset_index(drop=True), base[kept].reset_index(drop=True))
    assert not base["selected"].any()


def test_record_selection_stores_spec_rationale_timestamp_and_snapshot(tmp_path):
    grid = one_state_grid()
    tab = grid.comparison
    text = "Equal diagonal covariance matches the study design."
    chosen = int(tab.loc[tab["bic"].idxmax(), "model_id"])
    via_method = grid.record_selection(chosen, text)
    assert via_method.chosen["model_id"] == chosen
    assert not grid.comparison["selected"].any()
    selection = pf.record_selection(tab, chosen, text)
    assert selection.format_version == 1
    assert selection.rationale == text
    assert selection.timestamp.endswith("Z")
    assert selection.chosen["K"] == 1
    assert selection.chosen["covariance"] == tab.loc[tab["model_id"] == chosen, "covariance"].iloc[0]
    assert len(selection.comparison) == len(tab)
    assert int(selection.comparison["selected"].sum()) == 1
    assert not tab["selected"].any()
    assert set(tab["selection_rationale"]) == {""}
    path = tmp_path / "selection.json"
    selection.save(path)
    loaded = pf.load_model(path)
    assert loaded.rationale == text
    assert loaded.timestamp == selection.timestamp
    assert loaded.chosen["K"] == selection.chosen["K"]
    assert loaded.chosen["covariance"] == selection.chosen["covariance"]
    report = loaded.report()
    assert text in report
    assert selection.timestamp in report
    assert "K=1" in report
    assert selection.eligible_for_downstream["model_comparison"] is True
    assert selection.eligible_for_downstream["measurement"] is True
    assert selection.eligibility_detail == ""
    assert "Eligible for measurement." in selection.report(procedure="measurement")
    assert "No model has been selected" in pf.report_model(tab)
    with pytest.raises(ValueError, match="rationale"):
        pf.record_selection(tab, chosen, "   ")


def test_an_unreplicated_start_is_recorded_and_is_not_validated():
    fit = pf.fit_lpa(gaussian_sample(), ["y1", "y2"], n_states=1, covariance="VVV", n_starts=1, max_iter=40, seed=1)
    tab = pf.compare_models(fit)
    assert bool(tab["reliable"].iloc[0])
    assert tab["unreliable_reason"].iloc[0] == ""
    assert "replication_not_assessed" in tab["inference_warning"].iloc[0]
    selection = pf.record_selection(tab, 1, "A single start is not enough.")
    assert selection.rationale == "A single start is not enough."
    assert selection.eligible_for_downstream["model_comparison"] is False
    assert selection.eligible_for_downstream["measurement"] is True
    assert "replication_not_assessed" in selection.eligibility_detail
    assert "not an endorsement" in selection.report()
    with pytest.raises(ValueError, match="not eligible"):
        selection.report(procedure="model_comparison")


def test_default_grid_crosses_the_four_release_1_covariance_structures():
    default = inspect.signature(pf.fit_grid).parameters["covariance"].default
    assert default == ("EEI", "VVI", "EEE", "VVV")
    with pytest.raises(Exception, match="not in Release 1"):
        pf.fit_grid(gaussian_sample(), ["y1", "y2"], n_states=[1], covariance=["EII"], n_starts=1, max_iter=5)


def test_unsupported_covariance_is_not_aliased():
    with pytest.raises(Exception, match="not in Release 1"):
        pf.fit_grid(gaussian_sample(), ["y1", "y2"], n_states=[1], covariance=["EII"], n_starts=1, max_iter=5)


def test_categorical_grid_is_not_repeated_per_covariance():
    data = pd.DataFrame({"u": [0, 1] * 40, "v": [0, 0, 1, 1] * 20})
    grid = pf.fit_grid(
        data, ["u", "v"], model="lca", n_states=[1, 2],
        n_starts=2, max_iter=25, screen_iter=5, n_full=2, seed=1,
    )
    assert len(grid.comparison) == 2
    assert grid.comparison["covariance"].isna().all()
    assert not grid.comparison["selected"].any()


def test_enumerate_waves_is_diagnostic_per_wave():
    rng = np.random.default_rng(2)
    long = pd.DataFrame({
        "id": list(range(50)) * 2,
        "wave": [1] * 50 + [2] * 50,
        "y": np.concatenate([rng.normal(-0.4, size=50), rng.normal(0.5, size=50)]),
    })
    enum = pf.enumerate_waves(
        long, ["y"], id="id", wave="wave", n_states=[1, 2], covariance=["EEI"],
        n_starts=2, max_iter=30, screen_iter=5, n_full=2, seed=2,
    )
    assert set(enum.tables) == {1, 2}
    assert not hasattr(enum, "recommended_k")
    assert "Extra states can compensate for wrongly imposed invariance" in enum.note
    for table in enum.tables.values():
        assert len(table) == 2
        assert set(table["role"]) == {"wave_diagnostic"}
        assert "Extra states can compensate for wrongly imposed invariance" in table["diagnostic_note"].iloc[0]
    recorded = pf.record_selection(enum, 1, "Wave 1 looks like one profile.")
    assert recorded.eligible_for_downstream["model_comparison"] is False
    assert "wave_diagnostic" in recorded.eligibility_detail
    assert recorded.rationale == "Wave 1 looks like one profile."
    still = pf.record_selection(enum.tables[1], int(enum.tables[1]["model_id"].iloc[0]), "Still not joint.")
    assert still.eligible_for_downstream["model_comparison"] is False
    assert "wave_diagnostic" in still.eligibility_detail


def test_fit_criterion_plot_is_labelled_descriptive():
    grid = pf.fit_grid(
        gaussian_sample(), ["y1", "y2"], n_states=[1, 2], covariance=["VVV"],
        n_starts=2, max_iter=40, screen_iter=5, n_full=2, seed=1,
    )
    ax = pf.plot_fit_criteria(grid)
    title = ax.get_title()
    assert "Descriptive" in title
    assert "not a selection" in title
    assert "Entropy" in title


def _stub(**overrides):
    result = {
        "status_reason": "ok", "converged": True, "loglik": -10.0, "n_starts": 2, "n_best": 2,
        "regularized": False, "n_boundary": 0, "min_eigenvalue": float("nan"),
        "expected_membership": [10.0, 10.0], "expected_transitions": [],
        "ic_supported": True, "covariance": "", "K": 2,
        "measurement_invariant": True, "means_invariant": True, "covariance_invariant": True,
        "n_waves": 1, "transition_equal": True, "n_parameters": 3,
        "bic": 1.0, "sabic": 1.0, "aic": 1.0, "n_converged_starts": 2, "status": "ok",
        "n_analyzed": 20, "entropy_wave": 0.4, "entropy_sequence": 0.4, "gamma": [0.8, 0.2],
    }
    result.update(overrides)
    return pf.Fit(result=result, panel={}, spec={}, model="lca")


def test_numerical_degeneracy_blocks_downstream_inference():
    nonfinite = pf.compare_models(_stub(loglik=float("nan"), status="unusable", status_reason="numerical_failure", ic_supported=False))
    assert not bool(nonfinite["reliable"].iloc[0])
    assert "numerical_failure" in nonfinite["unreliable_reason"].iloc[0]
    kept = pf.record_selection(nonfinite, 1, "The likelihood is not finite and the choice stays on the record.")
    assert kept.eligible_for_downstream["assignment"] is False
    assert "numerical_failure" in kept.eligibility_detail
    with pytest.raises(ValueError, match="not eligible"):
        kept.report(procedure="assignment")
    near = pf.compare_models(_stub(min_eigenvalue=1e-9, covariance="VVV"))
    assert bool(near["reliable"].iloc[0])
    assert near["unreliable_reason"].iloc[0] == ""
    assert "near_singular" not in near["unreliable_reason"].iloc[0]
    assert "variance_floor" not in near["inference_warning"].iloc[0]


def test_numerically_empty_state_warns_and_limits_procedures():
    tab = pf.compare_models(_stub(
        status="warned", status_reason="numerically_empty_state", ic_supported=False,
        expected_membership=[20.0, 1e-12],
    ))
    assert bool(tab["reliable"].iloc[0])
    assert "numerically_empty_state" in tab["inference_warning"].iloc[0]
    assert "boundary" not in tab["inference_warning"].iloc[0]
    kept = pf.record_selection(tab, 1, "One state has essentially no mass. The choice stays on the record.")
    assert kept.chosen["model_id"] == 1
    assert kept.eligible_for_downstream["assignment"] is True
    assert kept.eligible_for_downstream["measurement"] is False
    assert kept.eligible_for_downstream["model_comparison"] is False
    assert "numerically_empty_state" in kept.eligibility_detail


def test_boundary_probability_warns_and_does_not_fail_the_fit():
    tab = pf.compare_models(_stub(n_boundary=2, status="warned", status_reason="boundary_parameters"))
    assert bool(tab["reliable"].iloc[0])
    assert tab["status"].iloc[0] == "warned"
    assert tab["unreliable_reason"].iloc[0] == ""
    assert "boundary_parameters" in tab["inference_warning"].iloc[0]
    assert "boundary parameter estimates" in tab["boundary_warning"].iloc[0]
    kept = pf.record_selection(tab, 1, "A response probability sits on the boundary. I am keeping this class solution.")
    assert kept.eligibility_detail == ""
    assert "boundary_parameters" in kept.eligibility_notes
    assert all(kept.eligible_for_downstream.values())
    assert "not an endorsement" in kept.report()
    assert "Eligible for measurement." in kept.report(procedure="measurement")


def test_variance_floor_warns_and_does_not_fail_the_fit():
    tab = pf.compare_models(_stub(
        regularized=True, min_eigenvalue=1e-12, covariance="EEI", ic_supported=False,
        status="warned", status_reason="variance_floor",
    ))
    assert bool(tab["reliable"].iloc[0])
    assert "near_singular" not in tab["unreliable_reason"].iloc[0]
    assert "variance_floor" in tab["inference_warning"].iloc[0]
    kept = pf.record_selection(tab, 1, "The covariance was ridged onto the variance floor.")
    assert kept.eligible_for_downstream["model_comparison"] is False
    assert kept.eligible_for_downstream["measurement"] is True
    assert "variance_floor" in kept.eligibility_detail


def test_problematic_choices_are_recorded_with_their_own_reasons():
    fits = [
        _stub(),
        _stub(n_starts=1, n_best=1, status="warned", status_reason="replication_not_assessed"),
        _stub(converged=False, n_best=0, status="unusable", status_reason="not_converged", ic_supported=False),
        _stub(status="warned", status_reason="numerically_empty_state", ic_supported=False),
        _stub(n_boundary=1, status="warned", status_reason="boundary_parameters"),
    ]
    tab = pf.compare_models(fits, role=["candidate", "candidate", "candidate", "candidate", "wave_diagnostic"])
    expected = [
        (1, True, ""),
        (2, False, "replication_not_assessed"),
        (3, False, "not_converged"),
        (4, False, "numerically_empty_state"),
        (5, False, "wave_diagnostic"),
    ]
    for model_id, comparison_ok, reason in expected:
        selection = pf.record_selection(tab, model_id, f"Keeping model {model_id} for the record.")
        assert selection.chosen["model_id"] == model_id
        assert selection.eligible_for_downstream["model_comparison"] is comparison_ok
        if reason:
            assert reason in selection.eligibility_detail
        if not comparison_ok:
            with pytest.raises(ValueError, match="not eligible"):
                pf.report_model(selection, procedure="model_comparison")
    boundary = pf.record_selection(tab, 5, "Boundary and diagnostic, still recorded.")
    assert "wave_diagnostic" in boundary.eligibility_detail
    assert boundary.eligible_for_downstream["measurement"] is True
    assert "boundary_parameters" in boundary.eligibility_notes
    clean = pf.record_selection(tab, 1, "This one replicated and did not hit a boundary.")
    assert "Eligible for measurement." in clean.report(procedure="measurement")
