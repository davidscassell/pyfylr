"""pyfylr: latent profile, latent class, and latent transition models.

Release 1 fits a first-order Markov latent transition model with the shared
C++ EM core. Indicators are Gaussian, Bernoulli, nominal, or a conditionally
independent product of one Gaussian block and categorical indicators.
Covariates are refused. Distal outcomes are Release 3. Ordinal indicators
are Release 2. Count families come after that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pyfylr import _core

__version__ = "0.1.0"
__all__ = [
    "Fit",
    "Grid",
    "Selection",
    "WaveEnumeration",
    "build_info",
    "accept",
    "add_starts",
    "align_states",
    "compare_invariance",
    "blrt_check",
    "blrt_last_telemetry",
    "blrt_failure_bounds",
    "blrt_hardware_threads",
    "blrt_null_size",
    "classification_diagnostics",
    "compare_models",
    "compare_transitions",
    "correlate_indicators",
    "decode_states",
    "describe_indicators",
    "describe_missingness",
    "describe_profiles",
    "enumerate_waves",
    "estimate_distal",
    "expected_counts",
    "fiml_moments",
    "fit_grid",
    "fit_lca",
    "fit_lpa",
    "fit_lta",
    "lca_spec",
    "lpa_spec",
    "lta_spec",
    "information_criteria",
    "plot_avepp",
    "plot_fit_criteria",
    "plot_ic_elbow",
    "plot_index_disagreement",
    "plot_max_posterior",
    "plot_prevalence",
    "plot_missingness",
    "plot_profiles",
    "plot_transitions",
    "posterior_states",
    "posterior_transitions",
    "load_model",
    "record_selection",
    "select_model",
    "selection_rule",
    "report_model",
    "save_model",
    "prepare_panel",
    "standardize_indicators",
    "state_prevalence",
    "transition_matrix",
    "LatentClassModel",
    "LatentProfileModel",
    "LatentTransitionModel",
    "StateCountSearch",
    "clone",
]


def build_info() -> dict:
    info = dict(_core.build_info())
    info["package"] = "pyfylr"
    info["version"] = __version__
    info["engine"] = "cpp-em"
    return info


def lta_spec(n_states, covariance="VVV", measurement_invariant=True, means_invariant=True,
             covariance_invariant=True, transitions="equal", n_starts=40, screen_iter=10,
             n_full=4, max_iter=200, tol=1e-7, replicate_tol=1e-3, n_threads=1, seed=1,
             align_labels=True, covariates=None, start=None, append_start=False):
    if covariates:
        raise ValueError(
            "Covariates are not estimated in Release 1. Pass indicators only. "
            "Covariates are a separate role and were not dropped into the measurement model."
        )
    if transitions not in ("equal", "free"):
        raise ValueError("transitions must be 'equal' or 'free'")
    spec = {
        "K": int(n_states),
        "covariance": None if covariance is None else _core.parse_covariance(str(covariance)),
        "measurement_invariant": bool(measurement_invariant),
        "means_invariant": bool(means_invariant),
        "covariance_invariant": bool(covariance_invariant),
        "transition_equal": transitions == "equal",
        "n_starts": int(n_starts),
        "screen_iter": int(screen_iter),
        "n_full": int(n_full),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "replicate_tol": float(replicate_tol),
        "n_threads": int(n_threads),
        "seed": int(seed),
        "align_labels": bool(align_labels),
    }
    if start is not None:
        spec["start"] = start
        if append_start:
            spec["append_supplied_start"] = True
        else:
            spec["has_start"] = True
            spec["n_starts"] = 1
    return spec


def lpa_spec(n_states, covariance="VVV", **kwargs):
    kwargs.setdefault("transitions", "equal")
    return lta_spec(n_states, covariance=covariance, **kwargs)


def lca_spec(n_states, **kwargs):
    kwargs["covariance"] = None
    return lta_spec(n_states, **kwargs)


def _family_of(series: pd.Series, declared: str | None) -> str:
    if declared is not None:
        return _core.parse_family(declared)
    numeric = pd.api.types.is_numeric_dtype(series)
    values = series.dropna().unique()
    if numeric and (len(values) > 2 or not set(np.round(values, 10)).issubset({0, 1})):
        return "gaussian"
    levels = series.dropna().unique()
    if len(levels) <= 2:
        return "bernoulli"
    return "nominal"


def _id_key(value):
    """Subject identity is exact. Wave codes may be rounded; subject ids may not."""
    if isinstance(value, (bool, np.bool_)):
        return ("s", str(value))
    if isinstance(value, (int, np.integer)):
        return ("n", int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ("s", str(value))
        number = float(value)
        if number.is_integer() and abs(number) < 2**53:
            return ("n", int(number))
        return ("f", number)
    return ("s", str(value))


def _wave_key(value):
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return ("n", int(round(float(value))))
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return ("n", int(value))
    return ("s", str(value))


def _looks_numeric(values):
    seq = list(values)
    if not seq:
        return False
    return all(isinstance(v, (int, np.integer, float, np.floating)) and not isinstance(v, (bool, np.bool_)) for v in seq)


def _integer_waves(values, what):
    """Numeric wave codes must be integers. Non-numeric labels are kept as given."""
    seq = list(values)
    if not seq:
        raise ValueError(f"{what} is empty")
    numeric = []
    non_numeric = False
    for value in seq:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{what} must be integer wave codes or an explicit non-numeric schedule")
        if isinstance(value, (int, np.integer)):
            numeric.append(int(value))
            continue
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value) or abs(float(value) - round(float(value))) > 1e-8:
                raise ValueError(
                    f"{what} must be integer wave codes; fractional schedules are rejected. "
                    "Pass an explicit schedule only when every entry is an integer."
                )
            numeric.append(int(round(float(value))))
            continue
        non_numeric = True
    if non_numeric:
        if numeric:
            raise ValueError(f"{what} mixes numeric and non-numeric wave codes")
        return seq
    return numeric


def _schema_from_panel(panel: dict) -> dict:
    return {
        "indicators": list(panel["indicators"]),
        "family": list(panel["family"]),
        "ncat": [int(v) for v in panel["ncat"]],
        "level_order": panel.get("level_order"),
        "waves": list(panel["waves"]),
        "measurement": panel["measurement"],
        "Rmax": int(panel["Rmax"]),
    }


def prepare_panel(data: pd.DataFrame, indicators: list[str], id: str, wave: str | None = None,
                  family: dict | list | None = None, schedule=None, schema: dict | None = None) -> dict:
    """Build a rectangular subject-by-scheduled-wave panel.

    Missing waves are inserted. They are not collapsed onto the next observed
    wave, so a gap still costs one transition per elapsed interval.

    ``schema`` is a training panel. Prediction reuses its category order,
    category counts, and wave schedule, and rejects a category that was not
    in that schema. Duplicate subject-wave keys are rejected. Automatic wave
    expansion accepts integer codes only.
    """
    if not indicators:
        raise ValueError("name at least one indicator")
    if id not in data.columns:
        raise ValueError(f"missing key column {id}")
    frame = data.copy()
    if wave is None:
        frame = frame.assign(_wave=1)
        wave = "_wave"
    if wave not in frame.columns:
        raise ValueError(f"missing key column {wave}")
    if frame[id].isna().any() or frame[wave].isna().any():
        raise ValueError("missing key in id or wave")
    dup = frame.duplicated(subset=[id, wave], keep=False)
    if dup.any():
        raise ValueError("duplicate subject-wave key")
    ids = pd.Index(frame[id].unique())
    if schema is not None:
        trained = _schema_from_panel(schema) if "indicators" in schema else schema
        if list(indicators) != list(trained["indicators"]):
            raise ValueError("prediction indicators do not match the training schema")
        waves = pd.Index(_integer_waves(trained["waves"], "training wave schedule")
                         if _looks_numeric(trained["waves"]) else list(trained["waves"]))
    elif schedule is not None:
        if len(list(schedule)) != len(set(map(_wave_key, schedule))):
            raise ValueError("duplicate schedule entry")
        waves = pd.Index(_integer_waves(schedule, "schedule"))
    else:
        observed = list(pd.unique(frame[wave]))
        if _looks_numeric(observed):
            codes = _integer_waves(observed, "wave")
            waves = pd.Index(range(min(codes), max(codes) + 1))
        else:
            # First appearance, the same rule as R. Lexical order is not a wave order.
            waves = pd.Index(list(dict.fromkeys(observed)))
    if schema is not None:
        trained = _schema_from_panel(schema) if "indicators" in schema else schema
        families = list(trained["family"])
    else:
        families = []
        for j, name in enumerate(indicators):
            declared = None
            if isinstance(family, dict):
                declared = family.get(name)
            elif isinstance(family, (list, tuple)):
                declared = family[j]
            families.append(_family_of(frame[name], declared))
    measurement = "mixed"
    if all(f == "gaussian" for f in families):
        measurement = "gaussian"
    elif all(f in ("bernoulli", "nominal") for f in families):
        measurement = "categorical"
    N, T, J = len(ids), len(waves), len(indicators)
    id_pos = {_id_key(v): i for i, v in enumerate(ids)}
    if len(id_pos) != N:
        raise ValueError("subject ids are not unique after identity normalization")
    wave_pos = {_wave_key(v): t for t, v in enumerate(list(waves))}
    gauss = np.full(N * T * J, np.nan)
    cat = np.full(N * T * J, -1, dtype=int)
    ncat = [0] * J
    levels = []
    level_order = []
    trained = _schema_from_panel(schema) if schema is not None and "indicators" in schema else schema
    for j, name in enumerate(indicators):
        if families[j] == "gaussian":
            levels.append(None)
            level_order.append(None)
            continue
        if trained is not None and trained.get("level_order"):
            order = list(trained["level_order"][j])
            ncat[j] = int(trained["ncat"][j])
            mapping = {}
            for code, label in enumerate(order):
                mapping[label] = code
                mapping[str(label)] = code
                if isinstance(label, (int, np.integer, float, np.floating)) and not isinstance(label, (bool, np.bool_)):
                    mapping[int(label)] = code
                    mapping[float(label)] = code
            levels.append(mapping)
            level_order.append(order)
            continue
        vals = pd.Series(frame[name].dropna().unique()).sort_values()
        numeric_vals = pd.to_numeric(vals, errors="coerce")
        coded_01 = numeric_vals.notna().all() and set(numeric_vals.round(10)).issubset({0.0, 1.0})
        if families[j] == "bernoulli" and coded_01:
            ncat[j] = 2
            levels.append({0: 0, 0.0: 0, 1: 1, 1.0: 1, "0": 0, "1": 1})
            level_order.append([0, 1])
            continue
        if families[j] == "bernoulli" and len(vals) != 2:
            raise ValueError(
                f"{name} was declared Bernoulli but does not have 2 observed categories. "
                "Code it 0/1, or pass both category labels in the data."
            )
        if len(vals) < 2:
            raise ValueError(f"{name} needs at least two observed categories to define a nominal indicator")
        ncat[j] = int(len(vals))
        order = [v.item() if hasattr(v, "item") else v for v in vals]
        levels.append({v: i for i, v in enumerate(order)})
        level_order.append(order)
    col = {name: i for i, name in enumerate(frame.columns)}
    numeric_schedule = _looks_numeric(list(waves))
    for rec in frame.itertuples(index=False, name=None):
        raw_wave = rec[col[wave]]
        if numeric_schedule:
            # Do not round a fractional code onto a neighboring wave.
            wave_key = ("n", int(_integer_waves([raw_wave], "wave")[0]))
        else:
            wave_key = _wave_key(raw_wave)
        if wave_key not in wave_pos:
            raise ValueError(f"wave {raw_wave!r} is not in the schedule")
        t = wave_pos[wave_key]
        i = id_pos[_id_key(rec[col[id]])]
        for j, name in enumerate(indicators):
            value = rec[col[name]]
            if pd.isna(value):
                continue
            offset = (i * T + t) * J + j
            if families[j] == "gaussian":
                gauss[offset] = float(value)
            else:
                key = value.item() if hasattr(value, "item") else value
                mapping = levels[j]
                if key not in mapping and str(key) not in mapping:
                    raise ValueError(f"unseen category {key!r} in {name}")
                cat[offset] = int(mapping[key] if key in mapping else mapping[str(key)])
    return {
        "N": N,
        "T": T,
        "J": J,
        "measurement": measurement,
        "family": families,
        "ncat": ncat,
        "Rmax": max(ncat + [1]),
        "cat": cat.tolist() if measurement != "gaussian" else None,
        "gauss": gauss.tolist() if measurement != "categorical" else None,
        "ids": list(ids),
        "waves": list(waves),
        "indicators": list(indicators),
        "levels": [{str(k): int(v) for k, v in lv.items()} if lv else None for lv in levels],
        "level_order": level_order,
    }


@dataclass
class Fit:
    result: dict
    panel: dict
    spec: dict
    model: str
    engine: str = "cpp-em"
    draws: object = None
    interval_type: str = "none"

    def glance(self) -> pd.DataFrame:
        r = self.result
        return pd.DataFrame([{
            "model": self.model,
            "status": r["status"],
            "status_reason": r["status_reason"],
            "engine": r["engine"],
            "loglik": r["loglik"],
            "aic": r["aic"],
            "bic": r["bic"],
            "sabic": r["sabic"],
            "n_parameters": r["n_parameters"],
            "n_analyzed": r["n_analyzed"],
            "n_excluded": r["n_excluded"],
            "n_states": r["K"],
            "n_waves": r["n_waves"],
            "entropy_wave": r["entropy_wave"],
            "entropy_sequence": r["entropy_sequence"],
            "converged": r["converged"],
            "n_best": r["n_best"],
            "n_starts": r["n_starts"],
            "ic_supported": r["ic_supported"],
            "interval_type": r["interval_type"],
            "parameter_uncertainty": r["parameter_uncertainty"],
            "state_uncertainty": r["state_uncertainty"],
            "has_draws": r["has_draws"],
            "covariance": r["covariance"],
            "measurement": r["measurement"],
        }])

    def tidy(self) -> pd.DataFrame:
        rows = []
        p = self.result["params"]
        K = self.result["K"]
        for k, value in enumerate(p["delta"]):
            rows.append({"parameter": "initial", "state": k + 1, "to": np.nan, "variable": np.nan,
                         "category": np.nan, "wave": np.nan, "estimate": value})
        if p["tau"]:
            blocks = len(p["tau"]) // (K * K)
            for b in range(blocks):
                for j in range(K):
                    for k in range(K):
                        rows.append({
                            "parameter": "transition", "state": j + 1, "to": k + 1,
                            "variable": np.nan, "category": np.nan,
                            "wave": np.nan if self.result["transition_equal"] else b + 1,
                            "estimate": p["tau"][(b * K + j) * K + k],
                        })
        names = self.panel["indicators"]
        if p["rho"]:
            J = self.panel["J"]
            R = self.result["Rmax"]
            blocks = len(p["rho"]) // (J * K * R)
            for b in range(blocks):
                for j, name in enumerate(names):
                    if self.panel["family"][j] == "gaussian":
                        continue
                    for k in range(K):
                        for r in range(self.panel["ncat"][j]):
                            rows.append({
                                "parameter": "response", "state": k + 1, "to": np.nan,
                                "variable": name, "category": r,
                                "wave": np.nan if self.result["measurement_invariant"] else b + 1,
                                "estimate": p["rho"][((b * J + j) * K + k) * R + r],
                            })
        if p["means"]:
            Jg = self.result["n_gaussian"]
            blocks = len(p["means"]) // (K * Jg)
            gidx = self.result["gaussian_index"]
            for b in range(blocks):
                for k in range(K):
                    for g, j in enumerate(gidx):
                        rows.append({
                            "parameter": "mean", "state": k + 1, "to": np.nan,
                            "variable": names[j], "category": np.nan,
                            "wave": np.nan if self.result["means_invariant"] else b + 1,
                            "estimate": p["means"][(b * K + k) * Jg + g],
                        })
        return pd.DataFrame(rows)

    def augment(self) -> pd.DataFrame:
        r = self.result
        N, T, K = r["n_subjects"], r["n_waves"], r["K"]
        rows = []
        for i, subject in enumerate(self.panel["ids"]):
            for t, wave in enumerate(self.panel["waves"]):
                base = (i * T + t) * K
                post = r["gamma"][base:base + K]
                filt = r["filtered"][base:base + K]
                rec = {
                    "id": subject,
                    "wave": wave,
                    "modal_state": None if r["modal"][i * T + t] < 0 else int(r["modal"][i * T + t]) + 1,
                    "viterbi_state": None if r["viterbi"][i * T + t] < 0 else int(r["viterbi"][i * T + t]) + 1,
                }
                for k in range(K):
                    rec[f"smoothed_{k + 1}"] = post[k]
                    rec[f"filtered_{k + 1}"] = filt[k]
                rows.append(rec)
        return pd.DataFrame(rows)

    def start_results(self) -> pd.DataFrame:
        r = self.result
        return pd.DataFrame({
            "start": r["start_id"],
            "loglik": r["start_loglik"],
            "iterations": r["start_iters"],
            "converged": r["start_converged"],
            "selected": r["start_selected"],
        })

    def likelihood_trace(self) -> pd.DataFrame:
        return pd.DataFrame({"iteration": self.result["trace_iter"], "loglik": self.result["trace_ll"]})


def _fit(data, indicators, id, wave, spec, model):
    panel = prepare_panel(data, indicators, id, wave, family=spec.get("family_map"))
    raw = dict(_core.fit(panel, spec))
    fit = Fit(result=raw, panel=panel, spec=spec, model=model, engine=raw["engine"],
              draws=None, interval_type=raw["interval_type"])
    return fit


def fit_lta(data, indicators, id, wave, n_states=2, family=None, covariance="VVV", **kwargs):
    spec = lta_spec(n_states, covariance=covariance, **kwargs)
    spec["family_map"] = family
    return _fit(data, list(indicators), id, wave, spec, "lta")


def fit_lpa(data, indicators, id=None, n_states=2, covariance="VVV", family=None, **kwargs):
    frame = data.copy()
    if id is None:
        frame = frame.assign(_id=np.arange(len(frame)))
        id = "_id"
    spec = lpa_spec(n_states, covariance=covariance, **kwargs)
    spec["family_map"] = family
    return _fit(frame, list(indicators), id, None, spec, "lpa")


def fit_lca(data, indicators, id=None, wave=None, n_states=2, family=None, **kwargs):
    frame = data.copy()
    if id is None:
        frame = frame.assign(_id=np.arange(len(frame)))
        id = "_id"
    spec = lca_spec(n_states, **kwargs)
    spec["family_map"] = family
    return _fit(frame, list(indicators), id, wave, spec, "lca")


def describe_indicators(data, indicators, id, wave=None, by_wave=False, family=None) -> pd.DataFrame:
    panel = prepare_panel(data, list(indicators), id, wave, family=family)
    rows = [dict(r) for r in _core.describe_indicators(panel, by_wave)]
    for row in rows:
        row["variable"] = panel["indicators"][row["variable"]]
        if row["wave"] is not None and row["wave"] >= 0:
            row["wave"] = panel["waves"][row["wave"]]
    return pd.DataFrame(rows)


def correlate_indicators(data, indicators, id=None, wave=None, family=None) -> pd.DataFrame:
    frame = data.copy()
    if id is None:
        frame = frame.assign(_id=np.arange(len(frame)))
        id = "_id"
    panel = prepare_panel(frame, list(indicators), id, wave, family=family)
    rows = []
    for raw in _core.correlate_indicators(panel):
        row = dict(raw)
        row["variable_a"] = panel["indicators"][row["variable_a"]]
        row["variable_b"] = panel["indicators"][row["variable_b"]]
        if row["wave"] is not None and row["wave"] < 0:
            row["wave"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fiml_moments(data, indicators, id=None, wave=None, covariance="VVV", family=None) -> dict:
    if covariance not in ("VVV", "EEI"):
        raise ValueError("fiml_moments accepts VVV (unrestricted) or EEI (diagonal)")
    frame = data.copy()
    if id is None:
        frame = frame.assign(_id=np.arange(len(frame)))
        id = "_id"
    panel = prepare_panel(frame, list(indicators), id, wave, family=family)
    moments = dict(_core.fiml_moments(panel, covariance))
    names = list(panel["indicators"])
    p = int(moments["p"])
    moments["mean"] = pd.Series(moments["mean"], index=names)
    moments["cov"] = pd.DataFrame(np.asarray(moments["cov"], dtype=float).reshape(p, p), index=names, columns=names)
    moments["cor"] = pd.DataFrame(np.asarray(moments["cor"], dtype=float).reshape(p, p), index=names, columns=names)
    return moments


def describe_missingness(data, indicators, id, wave=None, family=None) -> dict:
    panel = prepare_panel(data, list(indicators), id, wave, family=family)
    out = dict(_core.describe_missingness(panel))
    out["excluded_ids"] = [panel["ids"][i] for i in out["excluded_subjects"]]
    return out


def standardize_indicators(data, indicators, id, wave=None, scope="pooled") -> pd.DataFrame:
    if scope not in ("pooled", "within_wave"):
        raise ValueError("scope must be 'pooled' or 'within_wave'. The default is pooled.")
    panel = prepare_panel(data, list(indicators), id, wave)
    scaled = dict(_core.standardize_indicators(panel, scope))
    if scaled["warned"] and panel["measurement"] == "categorical":
        raise ValueError(scaled["warning"])
    out = data.copy()
    values = scaled["values"]
    out.attrs["center"] = list(scaled["center"]) if scaled["center"] is not None else []
    out.attrs["scale"] = list(scaled["scale"]) if scaled["scale"] is not None else []
    out.attrs["scope"] = scaled["scope"]
    out.attrs["warning"] = scaled["warning"]
    if values is None:
        return out
    # write standardized Gaussian columns back in id/wave order
    id_pos = {v: i for i, v in enumerate(panel["ids"])}
    wave_pos = {v: t for t, v in enumerate(panel["waves"])}
    wave_col = wave
    work = data.copy()
    if wave is None:
        work = work.assign(_wave=1)
        wave_col = "_wave"
        wave_pos = {1: 0}
    J = panel["J"]
    T = panel["T"]
    for idx, rec in work.iterrows():
        if rec[id] not in id_pos or rec[wave_col] not in wave_pos:
            continue
        i = id_pos[rec[id]]
        t = wave_pos[rec[wave_col]]
        for j, name in enumerate(panel["indicators"]):
            if panel["family"][j] != "gaussian":
                continue
            value = values[(i * T + t) * J + j]
            if value == value:
                out.at[idx, name] = value
    return out


def describe_profiles(fit: Fit, reference_sd=None) -> pd.DataFrame:
    rows = [dict(r) for r in _core.describe_profiles(fit.panel, fit.spec, fit.result, reference_sd)]
    for row in rows:
        row["variable"] = fit.panel["indicators"][row["variable"]]
        row["state"] = row["state"] + 1
    return pd.DataFrame(rows)


COMPARISON_COLUMNS = [
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
]

_ASSIGNMENT_NOTE = (
    "Entropy and the mean maximum posterior describe assignment certainty. "
    "They are not used to choose the number of states."
)
_BLRT_REASON = "BLRT is not validated in Release 1."
_LMR_REASON = "LMR/VLMR is not validated in Release 1."
_TRANSITION_NOTE = "Expected transition counts are reported. No sparsity cutoff is applied."


# Fit-level failures make the fit unusable. Everything else is a warning:
# the fit stays usable, and only some downstream procedures are ineligible.
# An absolute eigenvalue is not read. Entropy is not an input.
_BLOCKING = (
    "not_converged",
    "numerical_failure",
    "best_loglik_not_replicated",
    "not_positive_definite",
)
_WARNING = (
    "boundary_parameters",
    "variance_floor",
    "replication_not_assessed",
    "ill_conditioned_covariance",
    "collapsed_component",
    "numerically_empty_state",
)
_MEASUREMENT_BLOCK = ("not_positive_definite", "collapsed_component", "numerically_empty_state")
_COMPARISON_BLOCK = (
    "variance_floor",
    "replication_not_assessed",
    "not_positive_definite",
    "collapsed_component",
    "numerically_empty_state",
)
_NOTE_ONLY = ("boundary_parameters", "ill_conditioned_covariance")
_PROCEDURES = ("model_comparison", "assignment", "measurement", "transitions", "further_starts")


def _tokens(text) -> list[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    return [part for part in str(text).split(",") if part and part != "nan"]


def _split_reasons(result: dict) -> tuple[list[str], list[str]]:
    """Trust status_reason tokens. Do not re-derive a singularity cutoff."""
    tokens = set(_tokens(result.get("status_reason") or ""))
    blocking = []
    warning = []
    if not result.get("converged", True) or "not_converged" in tokens:
        blocking.append("not_converged")
    loglik = result.get("loglik")
    if loglik is None or not np.isfinite(loglik) or "numerical_failure" in tokens:
        blocking.append("numerical_failure")
    if (result.get("n_starts", 0) >= 2 and result.get("n_best", 0) < 2) or "best_loglik_not_replicated" in tokens:
        blocking.append("best_loglik_not_replicated")
    if "not_positive_definite" in tokens or result.get("cov_positive_definite") is False:
        blocking.append("not_positive_definite")
    for name in _WARNING:
        if name in tokens:
            warning.append(name)
    if result.get("n_starts", 0) < 2 or "replication_not_assessed" in tokens:
        warning.append("replication_not_assessed")
    if result.get("regularized") or "variance_floor" in tokens:
        warning.append("variance_floor")
    if (result.get("n_boundary") or 0) > 0 or "boundary_parameters" in tokens:
        warning.append("boundary_parameters")
    return list(dict.fromkeys(blocking)), list(dict.fromkeys(warning))


def _eligibility(blocking, warning, role) -> tuple[dict, str, str]:
    """Per-procedure eligibility.

    further_starts is always eligible. Assignment and transitions fail only
    on a fit-level block. Measurement also fails when the covariance is not
    positive definite, a component has collapsed, or a state is numerically
    empty. Model comparison fails on those, on a variance floor, on an
    unreplicated start, and on a per-wave diagnostic. Boundary estimates and
    ill-conditioning are notes: they do not remove eligibility.
    """
    block = set(blocking)
    warn = set(warning)
    flags = {name: True for name in _PROCEDURES}
    detail = []
    if block:
        primary = next(token for token in _BLOCKING if token in block)
        for proc in ("model_comparison", "assignment", "measurement", "transitions"):
            flags[proc] = False
            detail.append(f"{proc}:{primary}")
    else:
        meas = next((token for token in _MEASUREMENT_BLOCK if token in warn), None)
        if meas:
            flags["measurement"] = False
            detail.append(f"measurement:{meas}")
        comp = next((token for token in _COMPARISON_BLOCK if token in warn), None)
        if comp:
            flags["model_comparison"] = False
            detail.append(f"model_comparison:{comp}")
    if str(role) == "wave_diagnostic":
        flags["model_comparison"] = False
        detail = [item for item in detail if not item.startswith("model_comparison:")]
        detail.insert(0, "model_comparison:wave_diagnostic")
    order = {name: i for i, name in enumerate(("model_comparison", "assignment", "measurement", "transitions"))}
    detail.sort(key=lambda item: order.get(item.split(":", 1)[0], 9))
    notes = [token for token in _NOTE_ONLY if token in warn]
    return flags, ",".join(detail), ",".join(notes)


def _reliability(result: dict) -> tuple[bool, str, str]:
    blocking, warning = _split_reasons(result)
    return (len(blocking) == 0, ",".join(blocking), ",".join(warning))


def _mean_max_posterior(result: dict) -> float:
    gamma = np.asarray(result.get("gamma") or [], dtype=float)
    k = int(result["K"])
    if gamma.size == 0 or k < 1 or gamma.size % k:
        return np.nan
    mat = gamma.reshape(-1, k)
    keep = np.isfinite(mat).all(axis=1)
    if not keep.any():
        return np.nan
    return float(mat[keep].max(axis=1).mean())


def _array_equal(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        return False
    return bool(np.allclose(np.nan_to_num(aa, nan=0.0), np.nan_to_num(bb, nan=0.0), equal_nan=False) and np.array_equal(np.isnan(aa), np.isnan(bb)))


def _level_keys(order):
    if not order:
        return ()
    keys = []
    for item in order:
        if item is None:
            keys.append(None)
            continue
        row = []
        for value in item:
            if isinstance(value, (bool, np.bool_)):
                row.append(("s", str(value)))
            elif isinstance(value, (int, np.integer)):
                row.append(("n", int(value)))
            elif isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value) == round(float(value)):
                row.append(("n", int(round(float(value)))))
            else:
                row.append(("s", str(value)))
        keys.append(tuple(row))
    return tuple(keys)


def _panels_comparable(fits) -> bool:
    panels = []
    for fit in fits:
        panel = getattr(fit, "panel", None)
        if panel is None:
            return all(getattr(other, "panel", None) is None for other in fits)
        panels.append(panel)
    if not panels:
        return True
    base = panels[0]
    for panel in panels[1:]:
        if list(base.get("indicators") or []) != list(panel.get("indicators") or []):
            return False
        if list(base.get("family") or []) != list(panel.get("family") or []):
            return False
        if [_wave_key(w) for w in base.get("waves") or []] != [_wave_key(w) for w in panel.get("waves") or []]:
            return False
        if [_id_key(i) for i in base.get("ids") or []] != [_id_key(i) for i in panel.get("ids") or []]:
            return False
        if not _array_equal(base.get("gauss"), panel.get("gauss")):
            return False
        if not _array_equal(base.get("cat"), panel.get("cat")):
            return False
        if [int(v) for v in (base.get("ncat") or [])] != [int(v) for v in (panel.get("ncat") or [])]:
            return False
        if _level_keys(base.get("level_order")) != _level_keys(panel.get("level_order")):
            return False
    return True


def _criterion_note(frame: pd.DataFrame) -> str:
    return "BIC, AIC, and SABIC stay in separate columns. Nothing in this table totals them or chooses a model."


def _small_state(min_n, min_p, min_proportion, min_expected_n):
    if min_proportion is None and min_expected_n is None:
        return False, ""
    flagged = False
    parts = []
    if min_proportion is not None:
        parts.append(f"min_proportion={min_proportion}")
        if np.isfinite(min_p) and min_p < min_proportion:
            flagged = True
    if min_expected_n is not None:
        parts.append(f"min_expected_n={min_expected_n}")
        if np.isfinite(min_n) and min_n < min_expected_n:
            flagged = True
    return flagged, "; ".join(parts)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _plain(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return None if np.isnan(value) else value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def compare_models(fits, wave=None, role=None, min_proportion=None, min_expected_n=None) -> pd.DataFrame:
    if isinstance(fits, Grid):
        return fits.comparison.copy()
    if isinstance(fits, Fit):
        fits = [fits]
    fits = list(fits)
    if not fits:
        raise ValueError("compare_models needs at least one fit")
    n = len(fits)
    if wave is None:
        wave = [np.nan] * n
    if role is None:
        role = ["candidate"] * n
    if len(wave) != n or len(role) != n:
        raise ValueError("wave and role must have one entry per fit")
    rows = []
    for i, fit in enumerate(fits):
        result = fit.result
        reliable, blocked, warned = _reliability(result)
        membership = np.asarray(result.get("expected_membership") or [], dtype=float)
        transitions = np.asarray(result.get("expected_transitions") or [], dtype=float)
        min_n = float(membership.min()) if membership.size else np.nan
        min_p = min_n / result["n_analyzed"] if membership.size and result["n_analyzed"] else np.nan
        small_flag, small_rule = _small_state(min_n, min_p, min_proportion, min_expected_n)
        supported = bool(result["ic_supported"])
        covariance = result.get("covariance") or None
        rows.append({
            "model_id": i + 1,
            "role": role[i],
            "wave": wave[i],
            "model": fit.model,
            "K": result["K"],
            "covariance": covariance if covariance else np.nan,
            "measurement_invariant": result["measurement_invariant"],
            "means_invariant": result["means_invariant"],
            "covariance_invariant": result["covariance_invariant"],
            "transitions": np.nan if result["n_waves"] < 2 else ("equal" if result["transition_equal"] else "free"),
            "loglik": result["loglik"],
            "n_parameters": result["n_parameters"],
            "bic": result["bic"] if supported else np.nan,
            "sabic": result["sabic"] if supported else np.nan,
            "aic": result["aic"] if supported else np.nan,
            "ic_supported": supported,
            "ic_reason": "" if supported else "Information criteria are not supported for an unusable or regularized fit.",
            "n_best": result["n_best"],
            "n_starts": result["n_starts"],
            "n_converged_starts": result["n_converged_starts"],
            "converged": result["converged"],
            "status": result["status"],
            "min_expected_n": min_n,
            "min_expected_proportion": min_p,
            "small_state_flag": small_flag,
            "small_state_rule": small_rule,
            "min_expected_transitions": float(transitions.min()) if transitions.size else np.nan,
            "transition_note": _TRANSITION_NOTE,
            "entropy_wave": result["entropy_wave"],
            "entropy_sequence": result["entropy_sequence"],
            "mean_max_posterior": _mean_max_posterior(result),
            "assignment_note": _ASSIGNMENT_NOTE,
            "blrt": np.nan,
            "blrt_reason": _BLRT_REASON,
            "lmr": np.nan,
            "lmr_reason": _LMR_REASON,
            "n_boundary": result["n_boundary"],
            "regularized": result["regularized"],
            "min_eigenvalue": result["min_eigenvalue"],
            "boundary_warning": "boundary parameter estimates" if result["n_boundary"] > 0 else "",
            "regularization_warning": "covariance was ridged onto a variance floor; information criteria are not supported" if result["regularized"] else "",
            "reliable": reliable,
            "unreliable_reason": blocked,
            "inference_warning": warned,
            "criterion_note": "",
            "selected": False,
            "selection_rationale": "",
        })
    frame = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    note = _criterion_note(frame)
    frame["criterion_note"] = note
    same = _panels_comparable(fits)
    frame["comparable"] = same
    frame["incomparable"] = not same
    return frame


def _comparison_frame(comparison) -> pd.DataFrame:
    if isinstance(comparison, WaveEnumeration):
        return pd.concat(list(comparison.tables.values()), ignore_index=True)
    if isinstance(comparison, Grid):
        return comparison.comparison.copy()
    if isinstance(comparison, Selection):
        return comparison.comparison.copy()
    return comparison.copy()


def record_selection(comparison, chosen, rationale: str) -> Selection:
    """Store the analyst's choice. This does not endorse it.

    The record carries eligible_for_downstream, one flag per procedure.
    A boundary estimate stays eligible for every procedure. A variance floor
    blocks model comparison only. A per-wave row blocks model comparison
    with reason wave_diagnostic and leaves the other procedures as the fit
    left them.
    """
    frame = _comparison_frame(comparison)
    text = "" if rationale is None else str(rationale).strip()
    if not text:
        raise ValueError("Record the analyst's rationale. An empty rationale does not select a model.")
    hit = frame.index[frame["model_id"] == chosen].tolist()
    if not hit:
        raise ValueError("chosen is not a model_id in this comparison")
    snapshot = frame.copy()
    snapshot["selected"] = False
    snapshot["selection_rationale"] = ""
    snapshot.loc[hit, "selected"] = True
    snapshot.loc[hit, "selection_rationale"] = text
    chosen_row = snapshot.loc[hit[0]]
    flags, detail, notes = _eligibility(
        _tokens(chosen_row["unreliable_reason"]),
        _tokens(chosen_row["inference_warning"]),
        chosen_row["role"],
    )
    if len(hit) > 1:
        flags = dict(flags)
        flags["model_comparison"] = False
        extra = "model_comparison:model_id_not_unique_across_waves"
        detail = extra if not detail else detail + "," + extra
    spec = {
        "model_id": _plain(chosen_row["model_id"]),
        "model": _plain(chosen_row["model"]),
        "K": _plain(chosen_row["K"]),
        "covariance": _plain(chosen_row["covariance"]),
        "measurement_invariant": _plain(chosen_row["measurement_invariant"]),
        "means_invariant": _plain(chosen_row["means_invariant"]),
        "covariance_invariant": _plain(chosen_row["covariance_invariant"]),
        "transitions": _plain(chosen_row["transitions"]),
        "wave": _plain(chosen_row["wave"]),
        "role": _plain(chosen_row["role"]),
    }
    return Selection(
        chosen=spec, rationale=text, timestamp=_utc_now(),
        comparison=snapshot.reset_index(drop=True),
        eligible_for_downstream=flags, eligibility_detail=detail, eligibility_notes=notes,
    )


def _records(frame: pd.DataFrame) -> list:
    return json.loads(frame.to_json(orient="records"))


def save_model(obj, path) -> str:
    if isinstance(obj, Selection):
        payload = {
            "format_version": 1,
            "package": "pyfylr",
            "kind": "selection",
            "saved_at": _utc_now(),
            "chosen": obj.chosen,
            "rationale": obj.rationale,
            "timestamp": obj.timestamp,
            "eligible_for_downstream": obj.eligible_for_downstream,
            "eligibility_detail": obj.eligibility_detail,
            "eligibility_notes": obj.eligibility_notes,
            "comparison": _records(obj.comparison),
        }
    elif isinstance(obj, (Grid, pd.DataFrame)):
        frame = obj.comparison if isinstance(obj, Grid) else obj
        payload = {
            "format_version": 1,
            "package": "pyfylr",
            "kind": "comparison",
            "saved_at": _utc_now(),
            "chosen": None,
            "rationale": None,
            "timestamp": None,
            "comparison": _records(frame),
        }
    else:
        raise TypeError("save_model accepts a selection, a grid, or a comparison table")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return str(path)


def load_model(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format_version") != 1 or payload.get("package") != "pyfylr":
        raise ValueError("this file is not a pyfylr model archive of format version 1")
    frame = pd.DataFrame(payload["comparison"])
    if payload.get("kind") == "selection":
        return Selection(
            chosen=payload["chosen"],
            rationale=payload["rationale"],
            timestamp=payload["timestamp"],
            comparison=frame,
            format_version=1,
            eligible_for_downstream=dict(payload.get("eligible_for_downstream") or {}),
            eligibility_detail=payload.get("eligibility_detail") or "",
            eligibility_notes=payload.get("eligibility_notes") or "",
        )
    return frame


def report_model(obj, procedure: str | None = None) -> str:
    """Print a selection.

    With procedure set, this is a downstream use of that procedure and stops
    when that procedure is not eligible. With procedure omitted, every flag
    is printed and nothing is treated as a single pass/fail.
    """
    if type(obj).__name__ == "ModelSelection":
        return obj.reason + "\nThis is the automatic selection from select_model, not an analyst record. accepted starts unset."
    if isinstance(obj, Selection):
        flags = obj.eligible_for_downstream
        if procedure is not None:
            if procedure not in _PROCEDURES:
                raise ValueError("procedure must be one of " + ", ".join(_PROCEDURES))
            if not flags.get(procedure, False):
                why = obj.eligibility_detail or "unspecified"
                raise ValueError(f"This selection is not eligible for {procedure}: {why}")
        covariance = obj.chosen.get("covariance") or "none"
        lines = [
            "Analyst selection",
            f"Chosen spec: model={obj.chosen.get('model')}, K={obj.chosen.get('K')}, covariance={covariance}",
            f"Rationale: {obj.rationale}",
            f"Recorded: {obj.timestamp}",
            "Eligible for downstream use:",
        ]
        for name in _PROCEDURES:
            lines.append(f"  {name}: {'yes' if flags.get(name) else 'no'}")
        if obj.eligibility_detail:
            lines.append(f"Eligibility detail: {obj.eligibility_detail}")
        if obj.eligibility_notes:
            lines.append(f"Notes: {obj.eligibility_notes}")
        lines.extend([
            "This record preserves the analyst's choice. It is not an endorsement.",
            "The comparison table snapshot is stored with this selection.",
            "No automatic rule chose this model.",
        ])
        if procedure is not None:
            lines.append(f"Eligible for {procedure}.")
        return "\n".join(lines)
    return "No model has been selected. compare_models does not choose a number of states."


def _fit_state_grid(data, indicators, id, wave, n_states, covariance, family, longitudinal, kwargs):
    probe = data.copy()
    probe_id = id
    if probe_id is None:
        probe = probe.assign(_id=np.arange(len(probe)))
        probe_id = "_id"
    panel = prepare_panel(probe, list(indicators), probe_id, wave, family=family)
    gaussian = any(name == "gaussian" for name in panel["family"])
    covs = list(covariance) if gaussian else [None]
    if gaussian and not covs:
        raise ValueError("name at least one covariance structure")
    fits = []
    for k in n_states:
        for cov in covs:
            if longitudinal:
                fits.append(fit_lta(data, indicators, id=id, wave=wave, n_states=k, family=family, covariance=cov, **kwargs))
            elif gaussian:
                fits.append(fit_lpa(data, indicators, id=id, n_states=k, family=family, covariance=cov, **kwargs))
            else:
                fits.append(fit_lca(data, indicators, id=id, wave=wave, n_states=k, family=family, **kwargs))
    return fits


@dataclass
class Selection:
    chosen: dict
    rationale: str
    timestamp: str
    comparison: pd.DataFrame
    format_version: int = 1
    eligible_for_downstream: dict | None = None
    eligibility_detail: str = ""
    eligibility_notes: str = ""

    def report(self, procedure: str | None = None) -> str:
        return report_model(self, procedure=procedure)

    def save(self, path) -> str:
        return save_model(self, path)


@dataclass
class Grid:
    fits: list
    comparison: pd.DataFrame

    def record_selection(self, chosen, rationale: str) -> Selection:
        return record_selection(self.comparison, chosen, rationale)


@dataclass
class WaveEnumeration:
    tables: dict
    fits: list
    note: str


_WAVE_NOTE = (
    "Per-wave enumeration is diagnostic only. It does not choose the joint number of states. "
    "Extra states can compensate for wrongly imposed invariance."
)


def fit_grid(data, indicators, id=None, wave=None, n_states=range(1, 4),
             covariance=("EEI", "VVI", "EEE", "VVV"), model="detect", by_wave=False, family=None, **kwargs):
    if model not in ("detect", "lpa", "lca", "lta"):
        raise ValueError("model must be 'detect', 'lpa', 'lca', or 'lta'")
    states = [int(k) for k in n_states]
    if not states or any(k < 1 for k in states):
        raise ValueError("n_states must be one or more state counts, each at least 1")
    frame = data.copy()
    if by_wave or model == "lta":
        if wave is None or id is None:
            raise ValueError("a longitudinal grid needs id and wave")
    if by_wave:
        fits = []
        tags = []
        for w in sorted(frame[wave].unique(), key=lambda value: (str(type(value)), value)):
            part = frame.loc[frame[wave] == w].copy()
            piece = _fit_state_grid(part, indicators, id, None, states, covariance, family, False, kwargs)
            fits.extend(piece)
            tags.extend([w] * len(piece))
        comparison = compare_models(fits, wave=tags, role=["wave_diagnostic"] * len(fits))
        return Grid(fits=fits, comparison=comparison)
    longitudinal = model == "lta" or (model == "detect" and wave is not None and frame[wave].nunique() > 1)
    if longitudinal and (wave is None or id is None):
        raise ValueError("a joint latent transition grid needs id and wave")
    fits = _fit_state_grid(frame, indicators, id, wave if longitudinal else None, states, covariance, family, longitudinal, kwargs)
    comparison = compare_models(fits, role=["candidate"] * len(fits))
    return Grid(fits=fits, comparison=comparison)


def enumerate_waves(data, indicators, id, wave, **kwargs) -> WaveEnumeration:
    """Fit each wave on its own. Diagnostic only: this does not choose a joint number of states."""
    grid = fit_grid(data, indicators, id=id, wave=wave, by_wave=True, **kwargs)
    tables = {}
    for w, part in grid.comparison.groupby("wave", sort=True):
        table = part.copy()
        table["diagnostic_note"] = _WAVE_NOTE
        key = int(w) if isinstance(w, (int, np.integer)) else w
        tables[key] = table.reset_index(drop=True)
    return WaveEnumeration(tables=tables, fits=grid.fits, note=_WAVE_NOTE)


def plot_fit_criteria(comparison, ax=None):
    import matplotlib.pyplot as plt

    frame = comparison.comparison.copy() if isinstance(comparison, Grid) else comparison.copy()
    pieces = []
    for stat in ("bic", "sabic", "aic"):
        part = frame[["K", "covariance", "reliable", stat]].copy()
        part["criterion"] = stat.upper()
        part["value"] = part[stat]
        pieces.append(part)
    long = pd.concat(pieces, ignore_index=True)
    long = long[np.isfinite(long["value"])]
    if long.empty:
        raise ValueError("no supported information criteria to plot")
    long["covariance"] = long["covariance"].fillna("categorical").astype(str)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    for (cov, criterion), part in long.groupby(["covariance", "criterion"], sort=False):
        part = part.sort_values("K")
        ax.plot(part["K"], part["value"], label=f"{cov} {criterion}")
        reliable = part[part["reliable"]]
        unreliable = part[~part["reliable"]]
        if not reliable.empty:
            ax.scatter(reliable["K"], reliable["value"], marker="o")
        if not unreliable.empty:
            ax.scatter(unreliable["K"], unreliable["value"], marker="x")
    ax.set_title("Descriptive fit criteria\nNot a test and not a selection. Entropy is not used to choose the number of states.")
    ax.set_xlabel("States")
    ax.legend(fontsize=8)
    return ax


def compare_invariance(restricted: Fit, unrestricted: Fit) -> dict:
    return dict(_core.compare_invariance(restricted.panel, restricted.result, unrestricted.result))


def compare_transitions(equal_fit: Fit, free_fit: Fit) -> dict:
    return dict(_core.compare_transitions(equal_fit.panel, equal_fit.result, free_fit.result))


def decode_states(fit: Fit, method="smoothed") -> pd.DataFrame:
    aug = fit.augment()
    if method == "smoothed":
        return aug
    if method == "filtered":
        keep = [c for c in aug.columns if not c.startswith("smoothed_")]
        return aug[keep]
    if method == "viterbi":
        return aug[["id", "wave", "viterbi_state"]]
    raise ValueError("method must be 'smoothed', 'filtered', or 'viterbi'")


def estimate_distal(name, family, timing=None, wave=None, from_state=None, to_state=None, method=None):
    """Distal outcomes are Release 3. This call validates the declaration and stops."""
    _core.estimate_distal({
        "name": name,
        "family": family,
        "timing": timing,
        "wave": wave,
        "from_state": from_state,
        "to_state": to_state,
        "method": method,
    })


def plot_profiles(fit: Fit, ax=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    tidy = fit.tidy()
    means = tidy[tidy["parameter"] == "mean"]
    if means.empty:
        resp = tidy[tidy["parameter"] == "response"].copy()
        if resp.empty:
            raise ValueError("this fit has no profile parameters to plot")
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        sns.pointplot(data=resp, x="variable", y="estimate", hue="state", ax=ax)
        ax.set_ylabel("Category probability")
        ax.set_title("Response profiles")
        return ax
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    sns.pointplot(data=means, x="variable", y="estimate", hue="state", ax=ax)
    ax.set_ylabel("State mean")
    ax.set_title("Gaussian profiles")
    return ax


def plot_transitions(fit: Fit, ax=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    tidy = fit.tidy()
    trans = tidy[tidy["parameter"] == "transition"].copy()
    if trans.empty:
        raise ValueError("a one-wave model has no transition matrix")
    if trans["wave"].notna().any():
        trans = trans[trans["wave"] == trans["wave"].min()]
    mat = trans.pivot(index="state", columns="to", values="estimate")
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    sns.heatmap(mat, annot=True, vmin=0, vmax=1, cmap="Blues", ax=ax)
    ax.set_title("Transition probabilities")
    return ax


def _same_fit(result, fit: Fit) -> Fit:
    return Fit(result=result, panel=fit.panel, spec=fit.spec, model=fit.model,
               engine=result.get("engine", "cpp-em"), draws=None,
               interval_type=result.get("interval_type", "none"))


def add_starts(fit: Fit, n_new, seed=1) -> Fit:
    """Draw more random starts and keep the replicated best.

    The new streams are seeded from `seed` mixed with the number of starts
    already run, so they do not repeat the original streams.
    """
    if int(n_new) < 1:
        raise ValueError("n_new must be at least 1")
    spec = dict(fit.spec)
    spec["has_start"] = False
    raw = dict(_core.add_starts(fit.panel, spec, fit.result, int(n_new), float(seed)))
    return _same_fit(raw, fit)


def align_states(fit: Fit, reference):
    """Match states to a reference fit or to a parameter dict.

    The match is the Hungarian assignment on the measurement-profile
    distance. Sorting by prevalence is a display convention and is not this
    match. The permutation is applied to means, probabilities, transition
    rows and columns, and the posterior arrays.
    """
    params = reference.result["params"] if isinstance(reference, Fit) else reference
    out = _core.align_states(fit.panel, fit.spec, fit.result, params)
    aligned = _same_fit(dict(out["result"]), fit)
    return aligned, dict(out["alignment"])


def posterior_states(fit: Fit, kind="smoothed") -> pd.DataFrame:
    """Subject-wave state probabilities. `kind` is smoothed, filtered, or modal."""
    if kind not in ("smoothed", "filtered", "modal"):
        raise ValueError("kind must be 'smoothed', 'filtered', or 'modal'")
    aug = fit.augment()
    if kind == "modal":
        return aug[["id", "wave", "modal_state"]].rename(columns={"modal_state": "state"})
    prefix = "smoothed_" if kind == "smoothed" else "filtered_"
    cols = [c for c in aug.columns if c.startswith(prefix)]
    long = aug[["id", "wave"] + cols].melt(id_vars=["id", "wave"], var_name="state", value_name="probability")
    long["state"] = long["state"].str.replace(prefix, "", regex=False).astype(int)
    long["kind"] = kind
    return long.sort_values(["id", "wave", "state"]).reset_index(drop=True)


def posterior_transitions(fit: Fit) -> pd.DataFrame:
    """Adjacent-state joint probabilities (xi), one row per subject, interval, and pair."""
    result = fit.result
    k = int(result["K"])
    n = int(result["n_subjects"])
    t = int(result["n_waves"])
    xi = result.get("xi") or []
    columns = ["id", "from_wave", "from_state", "to_state", "probability"]
    if t < 2 or not xi:
        return pd.DataFrame(columns=columns)
    rows = []
    for i, subject in enumerate(fit.panel["ids"]):
        for step in range(t - 1):
            for a in range(k):
                for b in range(k):
                    rows.append({
                        "id": subject,
                        "from_wave": fit.panel["waves"][step],
                        "from_state": a + 1,
                        "to_state": b + 1,
                        "probability": xi[((i * (t - 1) + step) * k + a) * k + b],
                    })
    return pd.DataFrame(rows, columns=columns)


def transition_matrix(fit: Fit) -> pd.DataFrame:
    """Estimated transition probabilities, labelled by block when transitions are free."""
    tidy = fit.tidy()
    if tidy.empty:
        return tidy
    return tidy.loc[tidy["parameter"] == "transition"].reset_index(drop=True)


def state_prevalence(fit: Fit, kind="model") -> pd.DataFrame:
    """State mass by wave. `kind` is model, posterior, or hard."""
    key = {"model": "prevalence_model", "posterior": "prevalence_posterior", "hard": "prevalence_hard"}.get(kind)
    if key is None:
        raise ValueError("kind must be 'model', 'posterior', or 'hard'")
    values = list(fit.result[key])
    k = int(fit.result["K"])
    rows = []
    n_waves = len(values) // k if k else 0
    for t in range(n_waves):
        for state in range(k):
            rows.append({
                "kind": kind,
                "wave": fit.panel["waves"][t],
                "state": state + 1,
                "prevalence": values[t * k + state],
            })
    return pd.DataFrame(rows)


def expected_counts(fit: Fit) -> pd.DataFrame:
    """Expected state counts by wave and expected transition counts on each interval."""
    result = fit.result
    k = int(result["K"])
    membership = list(result["expected_membership"])
    rows = []
    n_waves = len(membership) // k if k else 0
    for t in range(n_waves):
        for state in range(k):
            rows.append({
                "quantity": "membership",
                "wave": fit.panel["waves"][t],
                "from_state": state + 1,
                "to_state": np.nan,
                "count": membership[t * k + state],
            })
    transitions = list(result.get("expected_transitions") or [])
    intervals = len(transitions) // (k * k) if k else 0
    for t in range(intervals):
        for a in range(k):
            for b in range(k):
                rows.append({
                    "quantity": "transition",
                    "wave": fit.panel["waves"][t],
                    "from_state": a + 1,
                    "to_state": b + 1,
                    "count": transitions[(t * k + a) * k + b],
                })
    return pd.DataFrame(rows)


def plot_missingness(data, indicators, id, wave=None, family=None, ax=None):
    import matplotlib.pyplot as plt

    summary = describe_missingness(data, indicators, id, wave, family)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    labels = list(summary["pattern"])
    counts = list(summary["pattern_n"])
    ax.barh(labels, counts, color="#1f4e79")
    ax.set_xlabel("Subjects")
    ax.set_title("Wave-missingness patterns")
    return ax


def __getattr__(name):
    if name == "pyfylr":
        return _accessor
    raise AttributeError(name)


@pd.api.extensions.register_dataframe_accessor("pyfylr")
class _PyfylrAccessor:
    def __init__(self, frame):
        self._frame = frame

    def fit_lpa(self, indicators, n_states=2, covariance="VVV", **kwargs):
        return fit_lpa(self._frame, indicators, n_states=n_states, covariance=covariance, **kwargs)

    def fit_lca(self, indicators, n_states=2, **kwargs):
        return fit_lca(self._frame, indicators, n_states=n_states, **kwargs)

    def fit_lta(self, indicators, id, wave, n_states=2, **kwargs):
        return fit_lta(self._frame, indicators, id, wave, n_states=n_states, **kwargs)


from pyfylr.selection import (  # noqa: E402
    accept,
    blrt_check,
    blrt_last_telemetry,
    blrt_failure_bounds,
    blrt_hardware_threads,
    blrt_null_size,
    classification_diagnostics,
    information_criteria,
    plot_avepp,
    plot_ic_elbow,
    plot_index_disagreement,
    plot_max_posterior,
    plot_prevalence,
    select_model,
    selection_rule,
)
from pyfylr.sklearn_api import (  # noqa: E402
    LatentClassModel,
    LatentProfileModel,
    LatentTransitionModel,
    StateCountSearch,
    clone,
)
