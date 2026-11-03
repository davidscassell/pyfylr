"""Automatic state-count and covariance selection.

`compare_models` stays a report. `select_model` applies the rule in
`selection_rule` when `auto` is true. The index-disagreement table records
which information criterion would pick which candidate. It is not a vote,
and `score` is not an input.

Information criteria, lower is better, with ``n`` the number of analyzed
subjects and ``EN`` the raw classification entropy (``0 log 0 = 0``):

- AIC = -2 ll + 2 p
- AIC3 = -2 ll + 3 p
- BIC = -2 ll + p log n
- SABIC = -2 ll + p log((n + 2) / 24)
- CAIC = -2 ll + p (log n + 1)
- AWE = -2 ll + 2 p (log n + 1.5) on the observed log-likelihood
- ICL-BIC = BIC + 2 EN

For one wave, EN is the entropy of the smoothed state. For a latent
transition model, EN is the joint sequence entropy: the entropy of the first
state plus the conditional entropy of each later state given the previous
one. Normalized entropy is reported and is not a default filter.

Deltas subtract the minimum of that index. The BIC-approximate Bayes factor
in favor of a model over the BIC-best model is exp(-0.5 * delta BIC).
Correct-model probabilities use those weights and sum to one.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

INDEX_NAMES = ("aic", "aic3", "bic", "sabic", "caic", "awe", "icl_bic")

BIAS_NOTE = {
    "aic": "AIC tends to over-select the number of states (Nylund, Asparouhov, and Muthen 2007).",
    "aic3": "AIC3 penalizes more than AIC and can still over-select relative to BIC (Bozdogan 1987; Nylund-Gibson and Choi 2018).",
    "bic": "BIC can under-select in small samples and for latent transition models (Nylund, Asparouhov, and Muthen 2007; Edelsbrunner, Flaig, and Schneider 2022).",
    "sabic": "SABIC can over-select relative to BIC in large cross-sectional samples (Nylund-Gibson and Choi 2018; Edelsbrunner, Flaig, and Schneider 2022).",
    "caic": "CAIC is more conservative than BIC and can under-select (Bozdogan 1987; Nylund-Gibson and Choi 2018).",
    "awe": "AWE is conservative and can under-select the number of states (Banfield and Raftery 1993; Nylund-Gibson and Choi 2018).",
    "icl_bic": "ICL-BIC adds a classification-entropy penalty and can under-select when classes overlap (Biernacki, Celeux, and Govaert 2000). It is a diagnostic and is not used for automatic ranking.",
}

HARD_TOKENS = (
    "hard:not_converged",
    "hard:non_finite",
    "hard:regularized",
    "hard:not_positive_definite",
    "hard:collapsed",
    "hard:replication_not_assessed",
    "hard:best_loglik_not_replicated",
    "hard:higher_admissible_not_replicated",
    "hard:information_criterion_undefined",
    "hard:second_seed_not_reproduced",
    "hard:second_seed_not_replicated",
    "hard:second_seed_not_assessed",
)
SOFT_TOKENS = (
    "soft:min_class_prop",
    "soft:min_class_n",
    "soft:min_avepp",
    "soft:entropy",
)
# The three second-seed tokens are applied inside the second-seed pass.
# They are not reasons to skip that pass.
_PRESEED = set(HARD_TOKENS[:-3])

CANDIDATE_COLUMNS = [
    "model_id", "model", "K", "covariance", "n_parameters", "n_waves", "n_analyzed", "loglik",
    "aic", "aic3", "bic", "sabic", "caic", "awe", "icl_bic",
    "delta_aic", "delta_aic3", "delta_bic", "delta_sabic", "delta_caic", "delta_awe", "delta_icl_bic",
    "bic_bf_vs_best", "bic_cmp",
    "min_class_prop", "min_class_n", "min_avepp",
    "entropy_wave", "entropy_sequence", "entropy_raw",
    "converged", "finite", "regularized", "positive_definite", "collapsed",
    "n_starts", "n_best", "replicate_tol",
    "second_seed_loglik", "second_seed_n_best", "second_seed_abs_gap", "second_seed_status",
    "hard_pass", "soft_pass", "eligible", "in_tie_zone",
    "rejection_reason", "selected",
]


def selection_rule(min_class_prop=None, min_class_n=None, min_avepp=None,
                   entropy_filter=False, min_entropy=0.80, primary=None, tie_zone=2,
                   second_seed_offset=10007, require_second_seed=True,
                   blrt=False, blrt_B=1000, blrt_max_retries=3, blrt_alpha=0.05,
                   blrt_check=False, blrt_max_iter=1000, experimental_blrt=False):
    """Default automatic rule. Substantive thresholds are off until supplied.

    ``primary=None`` means BIC. ICL-BIC is reported and is not a ranking
    criterion. Class size, AvePP, and entropy are shown and do not filter
    unless the analyst sets a threshold. BLRT defaults off and cannot change
    the selected model while ``validated`` is false. ``blrt_max_iter`` is the
    iteration cap for bootstrap fits only. Its default is 1000. Every other
    fit keeps ``max_iter``, whose default is 200. The tolerance stays 1e-7.
    """
    if primary == "icl_bic":
        raise ValueError("icl_bic is a diagnostic and is not used for automatic ranking")
    if primary is not None and primary not in INDEX_NAMES:
        raise ValueError("primary must be one of " + ", ".join(INDEX_NAMES))
    if tie_zone is None or float(tie_zone) < 0:
        raise ValueError("tie_zone must be zero or positive")
    if int(blrt_B) < 1:
        raise ValueError("blrt_B must be at least 1")
    if blrt_max_iter is None or int(blrt_max_iter) < 1:
        raise ValueError("blrt_max_iter must be at least 1")
    return {
        "min_class_prop": None if min_class_prop is None else float(min_class_prop),
        "min_class_n": None if min_class_n is None else float(min_class_n),
        "min_avepp": None if min_avepp is None else float(min_avepp),
        "entropy_filter": bool(entropy_filter),
        "min_entropy": None if min_entropy is None else float(min_entropy),
        "primary": primary,
        "tie_zone": float(tie_zone),
        "second_seed_offset": int(second_seed_offset),
        "require_second_seed": bool(require_second_seed),
        "blrt": bool(blrt),
        "blrt_B": int(blrt_B),
        "blrt_max_retries": int(blrt_max_retries),
        "blrt_alpha": float(blrt_alpha),
        "blrt_check": bool(blrt_check),
        "blrt_max_iter": int(blrt_max_iter),
        "experimental_blrt": bool(experimental_blrt),
    }


def _finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _cov_key(value):
    if value is None:
        return ""
    try:
        if np.isnan(value):
            return ""
    except TypeError:
        pass
    text = str(value)
    return "" if text in ("nan", "None", "") else text


def _cov_label(value):
    return _cov_key(value) or "categorical"


def _stored_entropy(result, name):
    if name not in result:
        return None
    value = result.get(name)
    if _finite(value):
        return float(value)
    return None


def raw_entropy(result):
    """Return (wave, sequence) raw entropies.

    Stored core values win. Otherwise the entropies are recomputed from the
    smoothed state and the pairwise joints with the same chain rule as the
    core. Non-positive mass contributes nothing.
    """
    stored_w = _stored_entropy(result, "entropy_wave_raw")
    stored_s = _stored_entropy(result, "entropy_sequence_raw")
    if stored_w is not None and stored_s is not None:
        return stored_w, stored_s
    gamma = np.asarray(result.get("gamma") or [], dtype=float)
    k = int(result.get("K") or 0)
    waves = int(result.get("n_waves") or 1)
    subjects = int(result.get("n_subjects") or 0)
    if k < 1 or gamma.size != subjects * waves * k:
        wave = stored_w if stored_w is not None else float("nan")
        sequence = stored_s if stored_s is not None else float("nan")
        return wave, sequence
    hw = 0.0
    hs = 0.0
    xi = np.asarray(result.get("xi") or [], dtype=float)
    for i in range(subjects):
        for t in range(waves):
            for state in range(k):
                g = gamma[(i * waves + t) * k + state]
                if np.isfinite(g) and g > 0:
                    hw -= g * np.log(g)
        for state in range(k):
            g = gamma[(i * waves + 0) * k + state]
            if np.isfinite(g) and g > 0:
                hs -= g * np.log(g)
        if waves < 2 or xi.size < subjects * (waves - 1) * k * k:
            continue
        for t in range(waves - 1):
            for j in range(k):
                g = gamma[(i * waves + t) * k + j]
                if not (np.isfinite(g) and g > 0):
                    continue
                for state in range(k):
                    x = xi[((i * (waves - 1) + t) * k + j) * k + state]
                    if np.isfinite(x) and x > 0:
                        hs -= x * np.log(x / g)
    if stored_w is not None:
        hw = stored_w
    if stored_s is not None:
        hs = stored_s
    return float(hw), float(hs)


def information_criteria(fit):
    """AIC, AIC3, BIC, SABIC, CAIC, AWE, and ICL-BIC for one fit.

    Blank when the fit does not support information criteria. ICL-BIC uses
    the joint sequence entropy when the fit has two or more waves.
    """
    result = fit.result if hasattr(fit, "result") else fit
    n = float(result.get("n_analyzed") or 0)
    p = float(result.get("n_parameters") or 0)
    ll = result.get("loglik")
    waves = int(result.get("n_waves") or 1)
    supported = bool(result.get("ic_supported"))
    hw, hs = raw_entropy(result)
    en = hs if waves >= 2 else hw
    out = {name: float("nan") for name in INDEX_NAMES}
    out.update({
        "n": n, "n_parameters": p, "loglik": float(ll) if _finite(ll) else float("nan"),
        "entropy_raw": float(en) if _finite(en) else float("nan"),
        "ic_supported": supported,
    })
    if not supported or not _finite(ll) or n < 1:
        return out
    dev = -2.0 * float(ll)
    logn = float(np.log(n))
    out["aic"] = dev + 2.0 * p
    out["aic3"] = dev + 3.0 * p
    out["bic"] = dev + p * logn
    out["sabic"] = dev + p * float(np.log((n + 2.0) / 24.0))
    out["caic"] = dev + p * (logn + 1.0)
    out["awe"] = dev + 2.0 * p * (logn + 1.5)
    if _finite(en):
        out["icl_bic"] = out["bic"] + 2.0 * float(en)
    return out


def _gamma_cube(result):
    gamma = np.asarray(result.get("gamma") or [], dtype=float)
    k = int(result.get("K") or 0)
    waves = int(result.get("n_waves") or 1)
    subjects = int(result.get("n_subjects") or 0)
    if k < 1 or gamma.size != subjects * waves * k:
        return None
    return gamma.reshape(subjects, waves, k)


def classification_diagnostics(fit):
    """AvePP, the full average-posterior matrix, OCC, class size, and entropy.

    AvePP for class k is the mean posterior of class k among person-waves
    whose modal class is k. The matrix also averages the other class
    posteriors in that modal group. OCC divides the posterior odds by the
    prevalence odds. A one-wave model has one block; a transition model also
    returns a block per wave.
    """
    result = fit.result if hasattr(fit, "result") else fit
    panel = getattr(fit, "panel", None) or {}
    k = int(result.get("K") or 0)
    waves = int(result.get("n_waves") or 1)
    cube = _gamma_cube(result)
    labels = list(panel.get("waves") or list(range(1, waves + 1)))
    if len(labels) != waves:
        labels = list(range(1, waves + 1))
    empty = {
        "avepp": [float("nan")] * k,
        "avepp_matrix": np.full((k, k), np.nan),
        "min_avepp": float("nan"),
        "occ": [float("nan")] * k,
        "min_class_prop": float("nan"),
        "min_class_n": float("nan"),
        "entropy_wave": result.get("entropy_wave", float("nan")),
        "entropy_sequence": result.get("entropy_sequence", float("nan")),
        "entropy_raw": raw_entropy(result)[1 if waves >= 2 else 0],
        "by_wave": pd.DataFrame(columns=["wave", "state", "avepp", "occ", "class_n", "class_prop", "n_modal"]),
    }
    membership = np.asarray(result.get("expected_membership") or [], dtype=float)
    n_analyzed = float(result.get("n_analyzed") or 0)
    mem = membership.reshape(waves, k) if membership.size == waves * k and k else None
    if mem is not None and n_analyzed > 0:
        empty["min_class_n"] = float(np.min(mem))
        empty["min_class_prop"] = float(np.min(mem / n_analyzed))
    if cube is None or k < 1:
        return empty

    def block(slc, prop_row):
        post = cube[:, slc, :] if not isinstance(slc, int) else cube[:, slc:slc + 1, :]
        post = post.reshape(-1, k)
        valid = np.isfinite(post).all(axis=1)
        modal = np.argmax(np.where(np.isfinite(post), post, -np.inf), axis=1)
        matrix = np.full((k, k), np.nan)
        avepp = np.full(k, np.nan)
        n_modal = np.zeros(k, dtype=int)
        for state in range(k):
            mask = valid & (modal == state)
            n_modal[state] = int(mask.sum())
            if n_modal[state] == 0:
                continue
            matrix[state, :] = post[mask].mean(axis=0)
            avepp[state] = matrix[state, state]
        occ = np.full(k, np.nan)
        if prop_row is not None:
            for state in range(k):
                occ[state] = _occ(avepp[state], prop_row[state])
        return avepp, matrix, occ, n_modal

    if mem is not None and n_analyzed > 0:
        prop = mem / n_analyzed
        pooled_prop = prop.mean(axis=0)
    else:
        prop = None
        pooled_prop = None
    avepp, matrix, occ, _n_modal = block(slice(None), pooled_prop)
    finite = avepp[np.isfinite(avepp)]
    rows = []
    for t in range(waves):
        prow = None if prop is None else prop[t]
        wave_ave, _wave_mat, wave_occ, wave_n = block(t, prow)
        for state in range(k):
            rows.append({
                "wave": labels[t],
                "state": state + 1,
                "avepp": wave_ave[state],
                "occ": wave_occ[state],
                "class_n": float("nan") if mem is None else float(mem[t, state]),
                "class_prop": float("nan") if prop is None else float(prop[t, state]),
                "n_modal": int(wave_n[state]),
            })
    hw, hs = raw_entropy(result)
    return {
        "avepp": [float(v) for v in avepp],
        "avepp_matrix": matrix,
        "min_avepp": float(finite.min()) if finite.size else float("nan"),
        "occ": [float(v) for v in occ],
        "min_class_prop": empty["min_class_prop"],
        "min_class_n": empty["min_class_n"],
        "entropy_wave": result.get("entropy_wave", float("nan")),
        "entropy_sequence": result.get("entropy_sequence", float("nan")),
        "entropy_raw": float(hs if waves >= 2 else hw),
        "by_wave": pd.DataFrame(rows),
    }


def _occ(avepp, pi):
    if not _finite(avepp) or not _finite(pi):
        return float("nan")
    if avepp <= 0 or pi <= 0 or pi >= 1:
        return float("nan")
    if avepp >= 1:
        return float("inf")
    return float((avepp / (1.0 - avepp)) / (pi / (1.0 - pi)))


def _soft_fail(value, threshold):
    if threshold is None:
        return False
    return (not _finite(value)) or float(value) < float(threshold)


def _primary_name(rule, rows):
    models = {row["model"] for row in rows}
    joint = any(row["model"] == "lta" and int(row["n_waves"]) >= 2 for row in rows)
    if joint and models != {"lta"}:
        raise ValueError("select_model compares one model family at a time")
    if rule["primary"]:
        return rule["primary"]
    return "bic"


def _join(tokens):
    ordered = [token for token in HARD_TOKENS + SOFT_TOKENS if token in tokens]
    return ",".join(ordered)


def _sort_key(row, primary):
    return (float(row[primary]), int(row["n_parameters"]), int(row["K"]), _cov_key(row["covariance"]), int(row["model_id"]))


def _choose(rows, primary, tie_zone):
    del tie_zone  # retained on the rule; a higher BIC is not preferred
    eligible = [row for row in rows if row["eligible"] and _finite(row[primary])]
    if not eligible:
        return None, False, []
    min_ic = min(float(row[primary]) for row in eligible)
    minimizers = [row for row in eligible if float(row[primary]) <= min_ic + 1e-8]
    pool = sorted(minimizers, key=lambda row: _sort_key(row, primary))
    return pool[0], len(minimizers) > 1, minimizers


def _disagreement(rows):
    records = []
    for name in INDEX_NAMES:
        finite = [row for row in rows if _finite(row[name])]
        if not finite:
            records.append({
                "index": name, "model_id": np.nan, "K": np.nan, "covariance": np.nan,
                "value": np.nan, "bias_note": BIAS_NOTE[name], "selects": False,
            })
            continue
        best = min(finite, key=lambda row: (float(row[name]), int(row["K"]), _cov_key(row["covariance"]), int(row["model_id"])))
        records.append({
            "index": name,
            "model_id": int(best["model_id"]),
            "K": int(best["K"]),
            "covariance": best["covariance"],
            "value": float(best[name]),
            "bias_note": BIAS_NOTE[name],
            "selects": False,
        })
    return pd.DataFrame(records, columns=["index", "model_id", "K", "covariance", "value", "bias_note", "selects"])


def _relative(rows):
    for name in INDEX_NAMES:
        finite = [float(row[name]) for row in rows if _finite(row[name])]
        baseline = min(finite) if finite else None
        for row in rows:
            row["delta_" + name] = float(row[name]) - baseline if baseline is not None and _finite(row[name]) else float("nan")
    weights = []
    for row in rows:
        if _finite(row["delta_bic"]):
            weights.append(float(np.exp(-0.5 * float(row["delta_bic"]))))
        else:
            weights.append(None)
    total = sum(w for w in weights if w is not None)
    for row, weight in zip(rows, weights):
        if weight is None or not total:
            row["bic_bf_vs_best"] = float("nan")
            row["bic_cmp"] = float("nan")
        else:
            row["bic_bf_vs_best"] = weight
            row["bic_cmp"] = weight / total


def _reason(decision, chosen, primary, rule, ambiguous, warnings, blrt):
    if decision == "auto_disabled":
        return (
            "Automatic selection is off because auto is FALSE. Candidates, rejection reasons, "
            "and the index disagreement table are reported. No cell is chosen."
        )
    if decision == "no_selection" or chosen is None:
        return (
            "No selection. No candidate passed the numerical validity checks. "
            "There is no fallback. See rejection_reason on each candidate."
        )
    cov = _cov_label(chosen["covariance"])
    value = float(chosen[primary])

    def setting(value):
        if value is None:
            return "NULL"
        if value is True:
            return "TRUE"
        if value is False:
            return "FALSE"
        number = float(value)
        if number.is_integer():
            return str(int(number))
        text = format(number, "g")
        return text

    def threshold_text(label, observed, threshold):
        shown = "NA" if not _finite(observed) else f"{float(observed):.6g}"
        if threshold is None:
            return f"{label} {shown} (no analyst threshold). "
        return f"{label} {shown} (analyst threshold {setting(threshold)} applied). "

    seed_text = (
        ", and an independent second seed reproduced that log-likelihood. "
        if rule["require_second_seed"] else ". The second-seed check is off. "
    )
    text = (
        f"Automatic selection: lowest {primary.upper()} among eligible candidates tested. "
        f"Selected the {chosen['model']} model with K={int(chosen['K'])} and covariance {cov} "
        f"({value:.6g}). "
        "Eligible means the fit converged, the likelihood is finite, the covariance is admissible "
        "(not regularized, positive definite, and not collapsed), and the highest admissible likelihood was replicated"
        f"{seed_text}"
        + threshold_text("Class size: minimum proportion", chosen["min_class_prop"], rule["min_class_prop"])
        + threshold_text("minimum class n", chosen["min_class_n"], rule["min_class_n"])
        + threshold_text("AvePP", chosen["min_avepp"], rule["min_avepp"])
    )
    entropy = chosen["entropy_wave"]
    entropy_shown = "NA" if not _finite(entropy) else f"{float(entropy):.6g}"
    if rule["entropy_filter"]:
        text += f"Entropy {entropy_shown} (analyst entropy threshold {setting(rule['min_entropy'])} applied). "
    else:
        text += f"Entropy {entropy_shown} is reported and is not a filter. "
    text += (
        f"Settings: auto=TRUE, primary={primary.upper()}, "
        f"min_class_prop={setting(rule['min_class_prop'])}, min_class_n={setting(rule['min_class_n'])}, "
        f"min_avepp={setting(rule['min_avepp'])}, entropy_filter={setting(bool(rule['entropy_filter']))}, "
        f"blrt={setting(bool(rule['blrt']))}, blrt_max_iter={setting(rule['blrt_max_iter'])}, "
        f"require_second_seed={setting(bool(rule['require_second_seed']))}. "
        "ICL-BIC is a diagnostic and is not used for automatic ranking. "
        "K and covariance were chosen jointly. "
    )
    if ambiguous:
        text += f"The choice is ambiguous: more than one eligible cell shares the minimum {primary.upper()}. "
    if "configural_noninvariance_possible" in warnings:
        text += " Per-wave enumeration selected different state counts, so configural noninvariance is possible."
    if blrt.get("ran") and not blrt.get("validated"):
        text += (
            " BLRT was run. validated stays false until Monte Carlo agreement, shared-dataset likelihoods, and operating characteristics are shown."
            " An unvalidated BLRT is report-only and does not change the selected model."
        )
    return text


def _panel_frame(fit):
    panel = fit.panel
    n = int(panel["N"])
    waves = int(panel["T"])
    j = int(panel["J"])
    rows = []
    for i in range(n):
        for t in range(waves):
            rec = {"id": panel["ids"][i], "wave": panel["waves"][t]}
            for col, name in enumerate(panel["indicators"]):
                offset = (i * waves + t) * j + col
                if panel["family"][col] == "gaussian":
                    value = panel["gauss"][offset]
                    rec[name] = value if _finite(value) else np.nan
                else:
                    code = int(panel["cat"][offset])
                    rec[name] = np.nan if code < 0 else code
            rows.append(rec)
    return pd.DataFrame(rows)


def _refit_second_seed(fit, offset):
    from pyfylr import _core

    spec = dict(fit.spec)
    spec["seed"] = int(spec.get("seed", 1)) + int(offset)
    spec["has_start"] = False
    spec.pop("start", None)
    return dict(_core.fit(fit.panel, spec))


def _wave_summary(fits, rule):
    from pyfylr import enumerate_waves

    fit = fits[0]
    if fit.panel is None:
        return None, "Per-wave enumeration was not run because the fits have no panel."
    frame = _panel_frame(fit)
    states = sorted({int(item.result["K"]) for item in fits})
    covs = []
    for item in fits:
        cov = item.result.get("covariance") or None
        if cov and cov not in covs:
            covs.append(cov)
    if not covs:
        covs = ["EEI"]
    spec = fit.spec
    family = {name: fam for name, fam in zip(fit.panel["indicators"], fit.panel["family"])}
    enum = enumerate_waves(
        frame, list(fit.panel["indicators"]), id="id", wave="wave",
        n_states=states, covariance=covs, family=family,
        n_starts=int(spec.get("n_starts", 2)),
        screen_iter=int(spec.get("screen_iter", 0)),
        n_full=int(spec.get("n_full", spec.get("n_starts", 2))),
        max_iter=int(spec.get("max_iter", 200)),
        tol=float(spec.get("tol", 1e-7)),
        replicate_tol=float(spec.get("replicate_tol", 1e-3)),
        seed=float(spec.get("seed", 1)),
        n_threads=int(spec.get("n_threads", 1)),
        align_labels=bool(spec.get("align_labels", True)),
    )
    wave_rule = dict(rule)
    wave_rule["primary"] = "bic"
    rows = []
    for wave, table in enum.tables.items():
        ids = [int(v) for v in table["model_id"].tolist()]
        piece = [enum.fits[i - 1] for i in ids]
        chosen = select_model(piece, rule=wave_rule, auto=True, waves="skip")
        rows.append({
            "wave": wave,
            "K": chosen.selected_k,
            "decision": chosen.decision,
            "model_id": chosen.selected_model_id,
        })
    summary = pd.DataFrame(rows)
    ks = [int(k) for k in summary["K"].dropna().tolist()]
    note = "Per-wave enumeration uses BIC. It does not choose the joint number of states."
    if len(set(ks)) > 1:
        note = "configural_noninvariance_possible"
    return summary, note


def _sigma(result, state, block=0):
    name = result.get("covariance") or ""
    params = result["params"]
    cov = np.asarray(params.get("cov") or [], dtype=float)
    k = int(result["K"])
    j = int(result["n_gaussian"])
    if j < 1 or cov.size == 0 or not name:
        return np.zeros((0, 0))
    blocks = max(int(result.get("n_cov_blocks") or 1), 1)

    def full(packed):
        matrix = np.zeros((j, j))
        index = 0
        for row in range(j):
            for col in range(row + 1):
                matrix[row, col] = matrix[col, row] = packed[index]
                index += 1
        return matrix

    if name == "EEI":
        start = block * j
        return np.diag(cov[start:start + j])
    if name == "VVI":
        start = (block * k + state) * j
        return np.diag(cov[start:start + j])
    packed = j * (j + 1) // 2
    if name == "EEE":
        start = block * packed
        return full(cov[start:start + packed])
    if name == "VVV":
        start = (block * k + state) * packed
        return full(cov[start:start + packed])
    return np.zeros((j, j))


def _masks(panel):
    n = int(panel["N"])
    waves = int(panel["T"])
    j = int(panel["J"])
    masks = np.zeros((n, waves, j), dtype=bool)
    for i in range(n):
        for t in range(waves):
            for col in range(j):
                offset = (i * waves + t) * j + col
                if panel["family"][col] == "gaussian":
                    masks[i, t, col] = not _finite(panel["gauss"][offset])
                else:
                    masks[i, t, col] = int(panel["cat"][offset]) < 0
    return masks


def _hash_columns(columns):
    digest = hashlib.sha256()
    n = 0
    for arr in columns.values():
        arr = np.asarray(arr)
        n += int(arr.size)
        if arr.dtype.kind in "biufc":
            digest.update(np.ascontiguousarray(arr).tobytes())
        else:
            digest.update(repr(arr.tolist()).encode())
    return f"{digest.hexdigest()}:{n}"


def _panel_from_columns(columns, panel):
    """Subject-major gaussian columns in the same order prepare_panel writes."""
    indicators = list(panel["indicators"])
    n = int(panel["N"])
    waves = int(panel["T"])
    j = int(panel["J"])
    stacked = np.column_stack([np.asarray(columns[name], dtype=np.float64) for name in indicators])
    return {
        "N": n,
        "T": waves,
        "J": j,
        "measurement": "gaussian",
        "family": [str(item) for item in panel["family"]],
        "ncat": [0] * j,
        "Rmax": 1,
        "cat": None,
        "gauss": np.ascontiguousarray(stacked).ravel(),
        "ids": list(range(1, n + 1)),
        "waves": list(panel["waves"]),
        "indicators": indicators,
        "levels": [None] * j,
    }


def _apply_gaussian(mu, factor, rows):
    # A diagonal factor matches one matrix product. A full factor is applied
    # one row at a time so the dots match the per-subject product.
    diagonal = np.diag(np.diag(factor))
    if np.array_equal(factor, diagonal):
        return mu + rows @ factor.T
    out = np.empty((rows.shape[0], rows.shape[1]), dtype=float)
    for i in range(rows.shape[0]):
        out[i] = mu + factor @ rows[i]
    return out


def _simulate_gaussian(result, panel, rng, donors, observed, delta, tau, means, k, waves, n, jg, gidx, as_panel=False):
    """Gaussian draw. Factors are built for this draw only, not for the whole bootstrap.

    One choice vector and one normal matrix match the per-row stream.
    """
    mean_blocks = max(int(result.get("n_mean_blocks") or 1), 1)
    cov_blocks = max(int(result.get("n_cov_blocks") or 1), 1)
    n_rows = n * waves
    if waves == 1:
        states = np.asarray(rng.choice(k, size=n, p=delta), dtype=np.int64)
    else:
        states = np.empty(n_rows, dtype=np.int64)
        for i in range(n):
            state = int(rng.choice(k, p=delta))
            states[i * waves] = state
            for t in range(waves - 1):
                block = 0 if result.get("transition_equal", True) else t
                probs = tau[(block * k + state) * k:(block * k + state + 1) * k]
                probs = np.clip(probs, 0, None)
                total = float(probs.sum())
                if total <= 0:
                    probs = np.full(k, 1.0 / k)
                else:
                    probs = probs / total
                state = int(rng.choice(k, p=probs))
                states[i * waves + t + 1] = state
    chol_cache = {}
    for cb in range(cov_blocks):
        for state in range(k):
            chol_cache[(state, cb)] = np.linalg.cholesky(_sigma(result, state, cb))
    z = rng.standard_normal((n_rows, jg))
    drawn = np.empty((n_rows, jg), dtype=float)
    t_index = np.tile(np.arange(waves), n)
    for state in range(k):
        state_rows = np.flatnonzero(states == state)
        if state_rows.size == 0:
            continue
        for cb in range(cov_blocks):
            if cov_blocks == 1:
                rows = state_rows
            else:
                rows = state_rows[np.minimum(t_index[state_rows], cov_blocks - 1) == cb]
            if rows.size == 0:
                continue
            factor = chol_cache[(state, cb)]
            if mean_blocks == 1:
                groups = [(0, rows)]
            else:
                mbs = np.minimum(t_index[rows], mean_blocks - 1)
                groups = [(int(mb), rows[mbs == mb]) for mb in np.unique(mbs)]
            for mb, sub in groups:
                if sub.size == 0:
                    continue
                mu = means[(mb * k + state) * jg:(mb * k + state + 1) * jg]
                drawn[sub] = _apply_gaussian(mu, factor, z[sub])
    indicators = list(panel["indicators"])
    value_cols = {name: np.full(n_rows, np.nan) for name in indicators}
    for g, index in enumerate(gidx):
        value_cols[indicators[int(index)]] = drawn[:, g]
    subj = np.repeat(np.arange(n), waves)
    tw = np.tile(np.arange(waves), n)
    donor = np.asarray(donors, dtype=np.int64)[subj]
    for col, name in enumerate(indicators):
        hit = observed[donor, tw, col]
        if np.any(hit):
            value_cols[name] = value_cols[name].copy()
            value_cols[name][hit] = np.nan
    wave_labels = list(panel["waves"])
    wave_col = np.empty(n_rows, dtype=object)
    if waves == 1:
        wave_col.fill(wave_labels[0])
    else:
        pattern = np.empty(waves, dtype=object)
        for t, label in enumerate(wave_labels):
            pattern[t] = label
        wave_col[:] = np.tile(pattern, n)
    out = {
        "id": np.repeat(np.arange(1, n + 1, dtype=np.int64), waves),
        "wave": wave_col,
    }
    out.update(value_cols)
    if as_panel:
        return _panel_from_columns(out, panel), _hash_columns(out)
    return pd.DataFrame(out)


def _simulate(fit, rng, donors, masks=None, as_panel=False):
    result = fit.result
    panel = fit.panel
    k = int(result["K"])
    waves = int(result["n_waves"])
    n = int(panel["N"])
    params = result["params"]
    delta = np.asarray(params["delta"], dtype=float)
    delta = delta / delta.sum()
    tau = np.asarray(params.get("tau") or [], dtype=float)
    means = np.asarray(params.get("means") or [], dtype=float)
    jg = int(result.get("n_gaussian") or 0)
    gidx = list(result.get("gaussian_index") or [])
    rho = np.asarray(params.get("rho") or [], dtype=float)
    observed = _masks(panel) if masks is None else masks
    if jg and rho.size == 0:
        return _simulate_gaussian(
            result, panel, rng, donors, observed, delta, tau, means, k, waves, n, jg, gidx,
            as_panel=as_panel,
        )
    mean_blocks = max(int(result.get("n_mean_blocks") or 1), 1)
    cov_blocks = max(int(result.get("n_cov_blocks") or 1), 1)
    paths = []
    for _i in range(n):
        state = int(rng.choice(k, p=delta))
        path = [state]
        for t in range(waves - 1):
            block = 0 if result.get("transition_equal", True) else t
            probs = tau[(block * k + state) * k:(block * k + state + 1) * k]
            probs = np.clip(probs, 0, None)
            if probs.sum() <= 0:
                probs = np.full(k, 1.0 / k)
            state = int(rng.choice(k, p=probs / probs.sum()))
            path.append(state)
        paths.append(path)
    indicators = list(panel["indicators"])
    families = list(panel["family"])
    rmax = int(result.get("Rmax") or 1)
    j = int(panel["J"])
    chol_cache = {}
    if jg:
        for cb in range(cov_blocks):
            for state in range(k):
                chol_cache[(state, cb)] = np.linalg.cholesky(_sigma(result, state, cb))
    id_col = np.empty(n * waves, dtype=int)
    wave_col = np.empty(n * waves, dtype=object)
    value_cols = {}
    for name, fam in zip(indicators, families):
        if fam == "gaussian":
            value_cols[name] = np.full(n * waves, np.nan, dtype=float)
        else:
            # Object buffers keep string category labels. A float buffer rejects them.
            column = np.empty(n * waves, dtype=object)
            column[:] = np.nan
            value_cols[name] = column
    cursor = 0
    for i in range(n):
        donor = int(donors[i])
        for t in range(waves):
            id_col[cursor] = i + 1
            wave_col[cursor] = panel["waves"][t]
            state = paths[i][t]
            if jg:
                mb = 0 if mean_blocks == 1 else min(t, mean_blocks - 1)
                cb = 0 if cov_blocks == 1 else min(t, cov_blocks - 1)
                mean = means[(mb * k + state) * jg:(mb * k + state + 1) * jg]
                draw = rng.standard_normal(jg)
                gauss = mean + chol_cache[(state, cb)] @ draw
                for g, index in enumerate(gidx):
                    value_cols[indicators[int(index)]][cursor] = float(gauss[g])
            if rho.size:
                mb = 0 if int(result.get("n_meas_blocks") or 1) == 1 else t
                for col, name in enumerate(indicators):
                    if families[col] == "gaussian":
                        continue
                    ncat = int(panel["ncat"][col])
                    probs = np.array([
                        rho[(((mb * j + col) * k + state) * rmax) + cat] for cat in range(ncat)
                    ], dtype=float)
                    probs = np.clip(probs, 0, None)
                    if probs.sum() <= 0:
                        probs = np.full(ncat, 1.0 / ncat)
                    code = int(rng.choice(ncat, p=probs / probs.sum()))
                    order = (panel.get("level_order") or [None] * (col + 1))[col]
                    if order is not None and code < len(order):
                        value_cols[name][cursor] = order[code]
                    else:
                        value_cols[name][cursor] = code
            mask = observed[donor, t]
            for col, name in enumerate(indicators):
                if mask[col]:
                    value_cols[name][cursor] = np.nan
            cursor += 1
    out = {"id": id_col, "wave": wave_col}
    out.update(value_cols)
    return pd.DataFrame(out)


def _split_null_state(source, k1):
    """Likelihood-preserving split of state 0, the same rule as the C++ core.

    Returns None when the stored parameter vectors do not match the layout.
    The caller then retries without this extra candidate.
    """
    from pyfylr import _core

    result = source.result
    params = dict(result.get("params") or {})
    k0 = int(result.get("K") or 0)
    if k0 < 1:
        return None
    panel = source.panel or {}
    cov = result.get("covariance") or "EEI"
    if not cov:
        cov = "EEI"
    tau = list(params.get("tau") or [])
    means = list(params.get("means") or [])
    if result.get("n_tau_blocks") is None:
        n_tau = 0 if not tau else len(tau) // (k0 * k0)
    else:
        n_tau = int(result["n_tau_blocks"])
    if result.get("gaussian_index") is None:
        n_mean = 1 if means else 0
        n_gaussian = len(means) // (n_mean * k0) if n_mean else 0
        n_cov = 1 if params.get("cov") else 0
    else:
        n_gaussian = len(result.get("gaussian_index") or [])
        n_mean = int(result.get("n_mean_blocks") or (1 if n_gaussian else 0))
        n_cov = int(result.get("n_cov_blocks") or (1 if n_gaussian else 0))
    rho = list(params.get("rho") or [])
    n_meas = int(result["n_meas_blocks"]) if result.get("n_meas_blocks") is not None else (1 if rho else 0)
    J = int(panel.get("J") or result.get("n_indicators") or 0)
    R = int(panel.get("Rmax") or result.get("Rmax") or 1)
    return _core.split_null_state(
        params, k0, int(k1), str(cov), int(n_tau), int(n_meas), int(n_mean), int(n_cov), int(n_gaussian), J, R
    )


def _fit_like(template, frame, seed, n_starts=None, n_full=None, max_iter=None, start=None, append_start=False):
    spec = template.spec
    result = template.result
    starts = max(int(spec.get("n_starts", 2)), 2) if n_starts is None else max(int(n_starts), 2)
    full = max(int(spec.get("n_full", spec.get("n_starts", 2))), 2) if n_full is None else max(int(n_full), 2)
    cap = int(spec.get("max_iter", 200)) if max_iter is None else int(max_iter)
    controls = {
        "n_starts": starts,
        "screen_iter": int(spec.get("screen_iter", 0)),
        "n_full": full,
        "max_iter": cap,
        "tol": float(spec.get("tol", 1e-7)),
        "replicate_tol": float(spec.get("replicate_tol", 1e-3)),
        "n_threads": int(spec.get("n_threads", 1)),
        "seed": int(seed),
        "align_labels": False,
    }
    if start is not None:
        controls["start"] = start
        if append_start:
            controls["append_start"] = True
    if isinstance(frame, dict) and "N" in frame and ("gauss" in frame or "cat" in frame):
        return _fit_prepared(template, frame, controls)
    prepared = _prepare_bootstrap_panel(template, frame)
    return _fit_prepared(template, prepared, controls)


def _fit_prepared(template, panel, controls):
    """Fit a rectangular panel. Do not pass it through the data-frame constructor."""
    from pyfylr import Fit, _core, lca_spec, lpa_spec, lta_spec

    spec_in = template.spec
    result = template.result
    model = template.model
    if model == "lta" or int(result["n_waves"]) >= 2:
        transitions = "equal" if spec_in.get("transition_equal", True) else "free"
        spec = lta_spec(
            int(result["K"]),
            covariance=result.get("covariance") or "EEI",
            transitions=transitions,
            measurement_invariant=bool(spec_in.get("measurement_invariant", True)),
            means_invariant=bool(spec_in.get("means_invariant", True)),
            covariance_invariant=bool(spec_in.get("covariance_invariant", True)),
            **controls,
        )
        model_name = "lta"
    elif model == "lca" or not result.get("covariance"):
        spec = lca_spec(int(result["K"]), **controls)
        model_name = "lca"
    else:
        spec = lpa_spec(int(result["K"]), covariance=result.get("covariance") or "EEI", **controls)
        model_name = model
    raw = dict(_core.fit(panel, spec))
    return Fit(
        result=raw, panel=panel, spec=spec, model=model_name, engine=raw["engine"],
        draws=None, interval_type=raw.get("interval_type", "none"),
    )


def _generating_schema(template):
    panel = template.panel
    return {
        "indicators": list(panel["indicators"]),
        "family": list(panel["family"]),
        "ncat": [int(v) for v in panel["ncat"]],
        "level_order": panel.get("level_order"),
        "waves": list(panel["waves"]),
        "measurement": panel.get("measurement"),
        "Rmax": int(panel.get("Rmax") or 1),
    }


def _prepare_bootstrap_panel(template, frame):
    """Prepare a simulated frame with the generating category counts and order."""
    from pyfylr import prepare_panel

    indicators = list(template.panel["indicators"])
    family = {name: fam for name, fam in zip(indicators, template.panel["family"])}
    wave = "wave" if template.model == "lta" or int(template.result["n_waves"]) >= 2 else None
    return prepare_panel(
        frame, indicators, "id", wave, family=family, schema=_generating_schema(template)
    )


def _blrt_solution(fit):
    """Likelihood, convergence, and replication of the returned start.

    A rejected start is not a substitute, even when its log-likelihood is higher.
    """
    result = fit.result
    returned = result.get("loglik")
    finite = returned is not None and np.isfinite(float(returned))
    converged = bool(result.get("converged"))
    if "admissible" in result and result.get("admissible") is not None:
        admissible = bool(result.get("admissible")) and finite and converged
    else:
        pd_flag = result.get("cov_positive_definite")
        admissible = (
            converged and finite and not bool(result.get("regularized"))
            and (pd_flag is None or bool(pd_flag))
            and not bool(result.get("cov_collapsed_component"))
        )
    n_best = int(result.get("n_best") or 0)
    return {
        "loglik": float(returned) if admissible else float("nan"),
        "converged": converged,
        "admissible": admissible,
        "replicated": bool(admissible and n_best >= 2),
        "n_best": n_best,
        "usable": admissible,
    }


def _higher_admissible_hidden(result):
    extra = result.get("rejected_loglik")
    if extra is None:
        return False
    values = np.atleast_1d(np.asarray(extra, dtype=float)).ravel()
    reasons = list(result.get("rejected_reason") or [])
    returned = result.get("loglik")
    if returned is None or not np.isfinite(float(returned)):
        return False
    blocked = ("not_converged", "numerical_failure", "not_positive_definite", "variance_floor", "collapsed_component")
    for i, value in enumerate(values):
        if not np.isfinite(value) or float(value) <= float(returned) + 1e-8:
            continue
        reason = reasons[i] if i < len(reasons) else ""
        if not any(token in str(reason) for token in blocked):
            return True
    return False


def _dataset_hash(frame):
    # Stable for a repeated draw. Not part of the likelihood.
    return _hash_columns({col: frame[col].to_numpy() for col in frame.columns})


def _wilson_ci(b, n, z=1.959963984540054):
    """Wilson interval. The lower end is 0 when b is 0, and the upper end is 1 when b is n."""
    if n is None or not np.isfinite(n) or int(n) < 1:
        return float("nan"), float("nan")
    n = float(n)
    phat = float(b) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = z * np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denom
    low = float(center - half)
    high = float(center + half)
    if low < 0.0 or int(b) == 0:
        low = 0.0
    if high > 1.0 or int(b) == int(n):
        high = 1.0
    return low, high


def _fmt_num(value):
    if value is None or not np.isfinite(value):
        return "NA"
    return format(float(value), ".6g")


def _blrt_summary(B, b, n_ok, n_failed, n_not_replicated, n_failed_not_converged,
                  n_failed_error, seconds, warn, failure_text):
    p = (b / n_ok) if n_ok else float("nan")
    p_plus = ((1 + b) / (1 + n_ok)) if n_ok else float("nan")
    mc_se = float(np.sqrt(p * (1.0 - p) / n_ok)) if n_ok else float("nan")
    ci_low, ci_high = _wilson_ci(b, n_ok)
    reason = (
        f"BLRT requested B={int(B)}. Kept {int(n_ok)} draws. "
        f"{int(n_not_replicated)} kept draws did not replicate the best log-likelihood. "
        f"Dropped {int(n_failed)} draws"
        f" (not converged: {int(n_failed_not_converged)},"
        f" error: {int(n_failed_error)}). "
        f"{failure_text} "
        f"Successful draws={int(n_ok)}. b={int(b)}. "
        f"Mplus TECH14 p=b/B={p:.6g} with B the successful draws. "
        f"p_plus=(b+1)/(B+1)={p_plus:.6g} is secondary and is not compared with alpha."
    )
    if warn:
        reason += (
            " THE BEST LOGLIKELIHOOD VALUE WAS NOT REPLICATED."
            " THE SOLUTION MAY NOT BE TRUSTWORTHY DUE TO LOCAL MAXIMA."
            " INCREASE THE NUMBER OF RANDOM STARTS."
        )
    reason += (
        " Unreplicated kept solutions are a diagnostic and are not a validation failure."
        f" Monte Carlo SE={_fmt_num(mc_se)}. Wilson 95% CI [{_fmt_num(ci_low)}, {_fmt_num(ci_high)}]."
        " Independent bootstrap p-values are not required to match."
        " validated stays false until Monte Carlo agreement, shared-dataset likelihoods, and operating characteristics are shown."
    )
    return p, p_plus, mc_se, ci_low, ci_high, reason


def blrt_null_size(alpha, B):
    """Null rejection probability of ``p = b/B`` when every draw is kept.

    Under H0 with continuous ranks the observed likelihood ratio is equally
    likely to sit in any of the ``B + 1`` positions. ``b/B <= alpha`` on
    ``floor(alpha * B) + 1`` of those positions.
    """
    alpha = float(alpha)
    draws = int(B)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if draws < 1:
        raise ValueError("B must be at least 1")
    return (int(np.floor(alpha * draws)) + 1) / (draws + 1)


def blrt_hardware_threads() -> int:
    """Hardware threads available to the draw pool. At least 1."""
    from pyfylr import _core

    return int(_core.blrt_hardware_threads())


def _blrt_fit_spec(template, n_starts, n_full, max_iter):
    from pyfylr import lca_spec, lpa_spec, lta_spec

    spec = template.spec
    result = template.result
    controls = {
        "n_starts": int(n_starts),
        "screen_iter": int(spec.get("screen_iter", 0)),
        "n_full": int(n_full),
        "max_iter": int(max_iter),
        "tol": float(spec.get("tol", 1e-7)),
        "replicate_tol": float(spec.get("replicate_tol", 1e-3)),
        "n_threads": 1,
        "seed": 1,
        "align_labels": False,
    }
    indicators = list(template.panel["indicators"])
    family = {name: fam for name, fam in zip(indicators, template.panel["family"])}
    model = template.model
    if model == "lta" or int(result["n_waves"]) >= 2:
        transitions = "equal" if spec.get("transition_equal", True) else "free"
        out = lta_spec(
            int(result["K"]),
            covariance=result.get("covariance") or "EEI",
            transitions=transitions,
            measurement_invariant=bool(spec.get("measurement_invariant", True)),
            means_invariant=bool(spec.get("means_invariant", True)),
            covariance_invariant=bool(spec.get("covariance_invariant", True)),
            **controls,
        )
    elif model == "lca" or not result.get("covariance"):
        out = lca_spec(int(result["K"]), **controls)
    else:
        out = lpa_spec(int(result["K"]), covariance=result.get("covariance") or "EEI", **controls)
    out["family_map"] = family
    return out


def _blrt_draw_pool(h0, h1, B, max_retries, base, cap, s0, f0, s1, f1, n_threads):
    from pyfylr import _core

    frames = []
    hashes = []
    present = []
    n = int(h0.panel["N"])
    masks = _masks(h0.panel)
    prep_t0 = time.perf_counter()
    for draw in range(int(B)):
        rng = np.random.default_rng(int(base) + 1000 * (draw + 1))
        donors = rng.integers(0, n, size=n)
        try:
            if str(h0.panel.get("measurement")) == "gaussian":
                built, digest = _simulate(h0, rng, donors, masks=masks, as_panel=True)
                frame = built
            else:
                frame = _simulate(h0, rng, donors, masks=masks)
                built = _prepare_bootstrap_panel(h0, frame)
                digest = _dataset_hash(frame)
        except (np.linalg.LinAlgError, ValueError):
            frame = None
        if frame is None:
            frames.append(None)
            hashes.append("")
            present.append(0)
            continue
        frames.append(built)
        hashes.append(digest)
        present.append(1)
    seconds_prepare = time.perf_counter() - prep_t0
    spec0 = _blrt_fit_spec(h0, s0, f0, cap)
    spec1 = _blrt_fit_spec(h1, s1, f1, cap)
    raw = _core.blrt_fit_draws(frames, present, spec0, spec1, int(base), int(max_retries), int(n_threads))
    asm_t0 = time.perf_counter()
    lrs = []
    lr_draws = []
    failures = []
    n_failed = n_failed_not_converged = n_failed_error = n_failed_inadmissible = 0
    n_not_replicated = 0
    n_retried = sum(1 for retry in raw["attempt_retry"] if int(retry) > 0)
    for draw, ok in enumerate(raw["ok"]):
        if not ok:
            why = str(raw["reason"][draw] or "error")
            n_failed += 1
            failures.append({"draw": draw + 1, "reason": why})
            if why == "not_converged":
                n_failed_not_converged += 1
            elif why == "inadmissible":
                n_failed_inadmissible += 1
            elif why == "h1_below_h0":
                n_failed_error += 1
            else:
                n_failed_error += 1
            continue
        lrs.append(float(raw["lr"][draw]))
        lr_draws.append(draw + 1)
        assessed = True if "h1_replication_assessed" not in raw else bool(raw["h1_replication_assessed"][draw])
        if not raw["h0_replicated"][draw] or (assessed and not raw["h1_replicated"][draw]):
            n_not_replicated += 1
    attempts = []
    h0_counts = raw.get("attempt_starts_h0", raw["attempt_starts"])
    h1_counts = raw.get("attempt_starts_h1", raw["attempt_starts"])
    for draw, retry, starts, h0_count, h1_count in zip(
        raw["attempt_draw"], raw["attempt_retry"], raw["attempt_starts"], h0_counts, h1_counts
    ):
        attempts.append({
            "draw": int(draw),
            "retry": int(retry),
            "dataset_hash": hashes[int(draw) - 1],
            "n_starts": int(starts),
            "n_starts_h0": int(h0_count),
            "n_starts_h1": int(h1_count),
        })
    jobs = [int(value) for value in raw["worker_jobs"]]
    busy = [float(value) for value in raw["worker_busy_s"]]
    if os.environ.get("LTA_PROFILE_WORKERS") and jobs:
        parts = " ".join(
            f"w{i}_jobs={jobs[i]} w{i}_busy_s={busy[i]:.6f}" for i in range(len(jobs))
        )
        print(f"worker_load n={len(jobs)} {parts}", flush=True)
    return {
        "lrs": lrs, "lr_draws": lr_draws, "failures": failures, "attempts": attempts,
        "n_failed": n_failed, "n_failed_not_converged": n_failed_not_converged,
        "n_failed_error": n_failed_error, "n_failed_inadmissible": n_failed_inadmissible,
        "n_not_replicated": n_not_replicated, "n_retried": n_retried,
        "n_threads_used": int(raw["n_threads_used"]),
        "worker_jobs": jobs, "worker_busy_s": busy,
        "seconds_prepare": seconds_prepare,
        "seconds_convert": float(raw["seconds_convert"]),
        "seconds_engine": float(raw["seconds_engine"]),
        "seconds_export": float(raw["seconds_export"]),
        "seconds_assemble": time.perf_counter() - asm_t0,
    }


# Same absolute gap the EM uses when it rejects a decrease. An alternative
# whose log-likelihood is lower than the null by more than this is not a
# successful bootstrap draw, and the likelihood ratio is not clamped to zero.
BLRT_MATERIAL_LL = 1e-5


def _h1_materially_below(ll0, ll1):
    """Same rule for the observed statistic and every bootstrap draw.

    A gap of exactly ``BLRT_MATERIAL_LL`` is kept. A larger negative gap is
    not a successful likelihood ratio and is not replaced by zero. The core
    uses the same absolute allowance in ``blrt_h1_materially_below``.
    """
    return bool(np.isfinite(ll0) and np.isfinite(ll1) and ll1 < ll0 - BLRT_MATERIAL_LL)


def _require_blrt_pair(h0, h1):
    from pyfylr import _panels_comparable

    if not _panels_comparable([h0, h1]):
        raise ValueError(
            "BLRT requires the same observed data, indicator encodings, wave schedule, and subjects"
        )
    if int(h0.result.get("K", 0)) >= int(h1.result.get("K", 0)):
        raise ValueError("BLRT requires a nested alternative with more states than the null")
    cov0 = h0.result.get("covariance") or ""
    cov1 = h1.result.get("covariance") or ""
    if cov0 != cov1:
        raise ValueError("BLRT requires the same covariance structure on the null and the alternative")
    for key in ("measurement_invariant", "means_invariant", "covariance_invariant", "transition_equal"):
        if bool(h0.spec.get(key, True)) != bool(h1.spec.get(key, True)):
            raise ValueError(f"BLRT requires the same {key} setting on the null and the alternative")


def blrt_failure_bounds(b, f, B, alpha=0.05, ran=True, n_not_run=0):
    """Plus-one bounds for an experimental BLRT over the scheduled draws.

    Exceedance is ``LR_b >= LR_obs``. ``b`` counts those exceedances among
    usable draws. ``f`` is the failed-draw count. ``n_not_run`` is the number
    of scheduled draws that were not attempted. Both are unresolved.
    ``n_unresolved = f + n_not_run``. ``B`` is the number of draws requested.
    ``p_lo = (b+1)/(B+1)`` counts unresolved draws as non-exceedances.
    ``p_hi = (b+n_unresolved+1)/(B+1)`` counts them as exceedances. When no
    draw was left unattempted this is ``(b+f+1)/(B+1)``.
    ``bounds_decision`` uses only that interval: reject when ``p_hi <= alpha``,
    retain when ``p_lo > alpha``, otherwise inconclusive. ``decision`` copies
    ``bounds_decision``, except that any unattempted draw forces
    ``decision`` to inconclusive. That force is an extra conservative rule.
    It is not the bounds decision. Any later BLRT operating-characteristic
    study scores the returned ``decision``, override included.
    ``bounds_decision`` stays separate and is not that study's rejection call.
    A check that did not run, or that
    requested no draws, returns ``not_run`` and leaves both bounds undefined.
    ``validated`` stays false. These bounds do not select a model.
    """
    b = int(b)
    f = int(f)
    requested = int(B)
    skipped = int(n_not_run)
    alpha = float(alpha)
    if b < 0 or f < 0 or requested < 0 or skipped < 0:
        raise ValueError("b, f, B, and n_not_run must be non-negative")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    unresolved = f + skipped
    out = {
        "b": b, "f": f, "B": requested, "n_not_run": skipped,
        "n_unresolved": unresolved, "alpha": alpha,
        "p_lo": float("nan"), "p_hi": float("nan"),
        "bounds_decision": "not_run", "decision": "not_run", "validated": False,
        "exceedance": "LR_b >= LR_obs",
    }
    if not ran or requested < 1:
        return out
    out["p_lo"] = (b + 1) / (requested + 1)
    out["p_hi"] = (b + unresolved + 1) / (requested + 1)
    if out["p_hi"] <= alpha:
        bounds_decision = "reject"
    elif out["p_lo"] > alpha:
        bounds_decision = "retain"
    else:
        bounds_decision = "inconclusive"
    out["bounds_decision"] = bounds_decision
    # Extra conservative rule, separate from the interval above.
    out["decision"] = "inconclusive" if skipped > 0 else bounds_decision
    return out


def _blrt_exceeds(lr_b, lr_obs):
    """Exceedance is LR_b >= LR_obs, with the locked 1e-8 tolerance."""
    return float(lr_b) >= float(lr_obs) - 1e-8


def _stamp_experimental_blrt(out):
    out["validated"] = False
    out["experimental"] = True
    out["switched"] = False
    failed = int(out.get("n_failed") or 0)
    kept = int(out.get("n_ok") or 0)
    requested = int(out.get("B") or 0)
    skipped = requested - kept - failed
    if skipped < 0:
        skipped = 0
    bounds = blrt_failure_bounds(
        int(out.get("b") or 0), failed, requested,
        float(out.get("alpha") or 0.05), ran=bool(out.get("ran")), n_not_run=skipped,
    )
    out["f"] = failed
    out["n_not_run"] = bounds["n_not_run"]
    out["n_unresolved"] = bounds["n_unresolved"]
    out["p_lo"] = bounds["p_lo"]
    out["p_hi"] = bounds["p_hi"]
    out["bounds_decision"] = bounds["bounds_decision"]
    out["decision"] = bounds["decision"]
    reason = str(out.get("reason") or "")
    if "blrt validated=false" not in reason:
        reason += " blrt validated=false."
    if out.get("ran"):
        reason += (
            f" Experimental bounds p_lo=(b+1)/(B+1)={bounds['p_lo']:.6g},"
            f" p_hi=(b+n_unresolved+1)/(B+1)={bounds['p_hi']:.6g}"
            f" with n_unresolved=f+n_not_run={bounds['n_unresolved']},"
            f" bounds_decision={bounds['bounds_decision']},"
            f" decision={bounds['decision']}."
            " Exceedance is LR_b >= LR_obs."
            " Unattempted draws are unresolved in the bounds over the scheduled B."
            " Any unattempted draw also forces decision=inconclusive."
            " That force is an extra conservative rule and is not the bounds decision."
            " These bounds do not change automatic selection."
        )
    elif bounds["decision"] == "not_run" and "not run" not in reason:
        reason = "not run: " + reason
    out["reason"] = reason
    out["report_only"] = True
    return out


_LAST_BLRT_TELEMETRY = {}


def blrt_last_telemetry():
    """Telemetry from the most recent ``blrt_check`` in this process.

    The mapping returned by ``blrt_check`` does not contain this record and
    has no telemetry attribute. A later call replaces it.
    """
    return dict(_LAST_BLRT_TELEMETRY)


def _blrt_publish(canonical, telemetry):
    global _LAST_BLRT_TELEMETRY
    _LAST_BLRT_TELEMETRY = dict(telemetry)
    return dict(canonical)


def blrt_check(h0, h1, B=1000, max_retries=3, alpha=0.05, seed=None, max_iter=None, n_threads=1,
               experimental=False):
    """Parametric bootstrap likelihood-ratio test of h0 against h1.

    ``p = b/B`` is reported. ``p_plus = (b+1)/(B+1)`` is secondary. The
    likelihood is the returned admissible converged start; a rejected start is
    never substituted. Each bootstrap dataset is simulated once, and a retry
    estimates that same dataset with more starts. ``validated`` stays false until
    Monte Carlo agreement, shared-dataset likelihoods, and operating
    characteristics are shown. Independent bootstrap p-values are not required
    to match. ``select_model`` does not let an unvalidated result change the
    selected model. The bootstrap does not run unless ``experimental`` is true.
    """
    if not experimental:
        skipped = {
            "ran": False, "validated": False, "experimental": False,
            "decision": "not_run", "report_only": True, "switched": False,
            "p": float("nan"), "p_plus": float("nan"),
            "p_lo": float("nan"), "p_hi": float("nan"),
            "b": 0, "f": 0, "B": int(B), "alpha": float(alpha),
            "reason": (
                "not run: experimental BLRT requires explicit opt-in (experimental=TRUE). "
                "blrt validated=false. Excluded from automatic selection."
            ),
        }
        print(skipped["reason"], flush=True)
        return _blrt_publish(skipped, {})
    s0 = max(int(h0.spec.get("n_starts", 2)), 2)
    s1 = max(int(h1.spec.get("n_starts", 2)), 2)
    f0 = max(int(h0.spec.get("n_full", s0)), 2)
    f1 = max(int(h1.spec.get("n_full", s1)), 2)
    runs = int(B) * (s0 + s1)
    cap = int(selection_rule()["blrt_max_iter"]) if max_iter is None else int(max_iter)
    if cap < 1:
        raise ValueError("blrt max_iter must be at least 1")
    n_threads = int(n_threads)
    if n_threads < 1:
        raise ValueError("n_threads must be at least 1")
    cost = (
        f"BLRT estimated cost: B={int(B)} draws x (H0 starts={s0} + H1 starts={s1}) "
        f"= {runs} EM runs before retries. Bootstrap fits use max_iter={cap}. "
        "Observed H0 and H1 fits keep their own max_iter. "
        "validated stays false until Monte Carlo agreement, shared-dataset likelihoods, and operating characteristics are shown."
    )
    print(cost, flush=True)
    sol0 = _blrt_solution(h0)
    sol1 = _blrt_solution(h1)
    out = {
        "ran": True, "validated": False, "cost_note": cost, "B": int(B),
        "starts_h0": s0, "starts_h1": s1, "estimated_runs": runs,
        "alpha": float(alpha), "p": float("nan"), "p_plus": float("nan"),
        "mc_se": float("nan"), "p_ci_low": float("nan"), "p_ci_high": float("nan"),
        "p_ci_method": "wilson_95",
        "b": 0, "lr_observed": 2.0 * (sol1["loglik"] - sol0["loglik"]),
        "observed_h0": sol0, "observed_h1": sol1,
        # replication_assessed means at least two starts were run. It does not
        # mean the best likelihood was found again. best_fit_replicated is that
        # second event, and observed_h1_replicated keeps the same value.
        "observed_h0_replicated": sol0["replicated"],
        "observed_h1_replicated": sol1["replicated"],
        "observed_h1_best_fit_replicated": sol1["replicated"],
        "observed_h1_replication_assessed": True,
        "n_ok": 0, "n_retried": 0, "n_failed": 0, "n_not_replicated": 0,
        "n_failed_not_converged": 0, "n_failed_error": 0, "n_failed_inadmissible": 0,
        "failures": [], "attempts": [],
        "switched": False, "report_only": True, "max_iter": cap,
        "n_threads": int(n_threads),
        "reason": "validated stays false until Monte Carlo agreement, shared-dataset likelihoods, and operating characteristics are shown.",
        "p_provisional": False,
    }
    if not sol0["usable"] or not sol1["usable"]:
        out["ran"] = False
        out["lr_observed"] = float("nan")
        out["reason"] = "not run: observed fit is not usable"
        print(out["reason"], flush=True)
        return _blrt_publish(_stamp_experimental_blrt(out), {})
    _require_blrt_pair(h0, h1)
    base = int(seed if seed is not None else h0.spec.get("seed", 1))
    if _h1_materially_below(sol0["loglik"], sol1["loglik"]):
        embedded = _split_null_state(h0, int(h1.result.get("K") or 0))
        if embedded is not None:
            refit = _fit_like(
                h1, h0.panel, base + 17, n_starts=s1, n_full=f1, max_iter=cap, start=embedded, append_start=True
            )
            sol1 = _blrt_solution(refit)
            assessed = int(refit.result.get("n_starts") or 0) >= 2
            out["observed_h1"] = sol1
            out["observed_h1_replication_assessed"] = assessed
            out["observed_h1_replicated"] = assessed and bool(sol1["replicated"])
            out["observed_h1_best_fit_replicated"] = out["observed_h1_replicated"]
            out["lr_observed"] = 2.0 * (sol1["loglik"] - sol0["loglik"])
        if not sol1["usable"] or _h1_materially_below(sol0["loglik"], sol1["loglik"]):
            out["ran"] = False
            out["lr_observed"] = float("nan")
            out["reason"] = "not run: observed alternative is materially below the null"
            print(out["reason"], flush=True)
            return _blrt_publish(_stamp_experimental_blrt(out), {})
    started = time.perf_counter()
    telemetry = {"n_threads_used": 1}
    if n_threads > 1:
        pooled = _blrt_draw_pool(h0, h1, B, max_retries, base, cap, s0, f0, s1, f1, n_threads)
        lrs = pooled["lrs"]
        lr_draws = pooled["lr_draws"]
        out["failures"] = pooled["failures"]
        out["attempts"] = pooled["attempts"]
        out["n_failed"] = pooled["n_failed"]
        out["n_failed_not_converged"] = pooled["n_failed_not_converged"]
        out["n_failed_error"] = pooled["n_failed_error"]
        out["n_failed_inadmissible"] = pooled["n_failed_inadmissible"]
        out["n_not_replicated"] = pooled["n_not_replicated"]
        out["n_retried"] = pooled["n_retried"]
        telemetry["n_threads_used"] = int(pooled["n_threads_used"])
        telemetry["worker_jobs"] = pooled["worker_jobs"]
        telemetry["worker_busy_s"] = pooled["worker_busy_s"]
        for key in ("seconds_prepare", "seconds_convert", "seconds_engine", "seconds_export", "seconds_assemble"):
            telemetry[key] = pooled[key]
    else:
        lrs = []
        lr_draws = []
        for draw in range(int(B)):
            ok = False
            why = "error"
            rng = np.random.default_rng(base + 1000 * (draw + 1))
            masks = _masks(h0.panel)
            donors = rng.integers(0, masks.shape[0], size=masks.shape[0])
            try:
                frame = _simulate(h0, rng, donors)
            except (np.linalg.LinAlgError, ValueError):
                frame = None
            if frame is None:
                out["failures"].append({"draw": draw + 1, "reason": "error"})
                out["n_failed"] += 1
                out["n_failed_error"] += 1
                continue
            dataset_hash = _dataset_hash(frame)
            embed_next = None
            for retry in range(int(max_retries) + 1):
                if retry:
                    out["n_retried"] += 1
                scale = retry + 1
                h1_start = embed_next
                embed_next = None
                embedded_this = h1_start is not None
                h1_count = s1 * scale + (1 if embedded_this else 0)
                out["attempts"].append({
                    "draw": draw + 1, "retry": retry, "dataset_hash": dataset_hash,
                    "n_starts": s0 * scale, "n_starts_h0": s0 * scale, "n_starts_h1": h1_count,
                })
                try:
                    fit0 = _fit_like(h0, frame, base + 1000 * (draw + 1) + retry, n_starts=s0 * scale, n_full=f0 * scale, max_iter=cap)
                    h1_call = {"n_starts": s1 * scale, "n_full": f1 * scale, "max_iter": cap}
                    if h1_start is not None:
                        h1_call["start"] = h1_start
                        h1_call["append_start"] = True
                    fit1 = _fit_like(h1, frame, base + 200000 + 1000 * (draw + 1) + retry, **h1_call)
                except (np.linalg.LinAlgError, ValueError):
                    why = "error"
                    continue
                draw0 = _blrt_solution(fit0)
                draw1 = _blrt_solution(fit1)
                if not draw0["usable"] or not draw1["usable"]:
                    why = "not_converged" if (not draw0["converged"] or not draw1["converged"]) else "inadmissible"
                    continue
                if _h1_materially_below(draw0["loglik"], draw1["loglik"]):
                    why = "h1_below_h0"
                    if retry < int(max_retries):
                        embed_next = _split_null_state(fit0, int(fit1.result.get("K") or 0))
                    continue
                lrs.append(2.0 * (draw1["loglik"] - draw0["loglik"]))
                lr_draws.append(draw + 1)
                assessed = h1_count >= 2
                if not draw0["replicated"] or (assessed and not draw1["replicated"]):
                    out["n_not_replicated"] += 1
                ok = True
                break
            if not ok:
                out["n_failed"] += 1
                out["failures"].append({"draw": draw + 1, "reason": why})
                if why == "not_converged":
                    out["n_failed_not_converged"] += 1
                elif why == "inadmissible":
                    out["n_failed_inadmissible"] += 1
                else:
                    out["n_failed_error"] += 1
    out["n_ok"] = len(lrs)
    out["lrs"] = lrs
    out["lr_draws"] = lr_draws
    out["b"] = sum(1 for value in lrs if _blrt_exceeds(value, out["lr_observed"]))
    telemetry["seconds"] = float(time.perf_counter() - started)
    if out["failures"]:
        failure_text = "Unresolved failures: " + "; ".join(
            f"draw {item['draw']} {item['reason']}" for item in out["failures"]
        ) + "."
    else:
        failure_text = "Unresolved failures: none."
    warn = (
        out["n_not_replicated"] > 0 or not sol0["replicated"]
        or (out["observed_h1_replication_assessed"] and not sol1["replicated"])
    )
    out["p_provisional"] = bool(out["n_failed"] > 0)
    out["p"], out["p_plus"], out["mc_se"], out["p_ci_low"], out["p_ci_high"], out["reason"] = _blrt_summary(
        out["B"], out["b"], out["n_ok"], out["n_failed"], out["n_not_replicated"],
        out["n_failed_not_converged"], out["n_failed_error"], telemetry["seconds"], warn, failure_text,
    )
    if out["p_provisional"]:
        out["reason"] += " p = b/n_ok is provisional because n_failed > 0."
    out = _stamp_experimental_blrt(out)
    print(out["reason"], flush=True)
    return _blrt_publish(out, telemetry)


class ModelSelection:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def selected_k(self):
        if self.decision != "selected":
            return None
        return int(self.candidates.loc[self.candidates["selected"], "K"].iloc[0])


def _candidate_frame(rows):
    frame = pd.DataFrame(rows)
    for column in CANDIDATE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[CANDIDATE_COLUMNS]


def select_model(fits, rule=None, auto=True, second_seed_loglik=None, second_seed_n_best=None, waves="auto"):
    """Choose K and covariance jointly. ``compare_models`` is not used to choose.

    Hard filters are convergence, a finite likelihood, an admissible covariance,
    and replication of the highest admissible likelihood, including the second
    seed. Class size, AvePP, and entropy are reported. They filter only when
    the analyst supplies a threshold. The primary index is BIC. ICL-BIC is a
    diagnostic. An unvalidated BLRT does not change the selected model.
    """
    from pyfylr import Grid, WaveEnumeration, compare_models

    if isinstance(fits, WaveEnumeration):
        raise ValueError(
            "per-wave enumeration does not choose the joint number of states. "
            "Pass the joint fits to select_model."
        )
    if isinstance(fits, Grid):
        fits = list(fits.fits)
    elif hasattr(fits, "result") and hasattr(fits, "model"):
        fits = [fits]
    else:
        fits = list(fits)
    if not fits:
        raise ValueError("select_model needs at least one fit")
    from pyfylr import _panels_comparable

    if auto and not _panels_comparable(fits):
        raise ValueError(
            "fits are not comparable: automatic ranking needs the same observations and response schema"
        )
    rule = dict(rule or selection_rule())
    supplied = second_seed_loglik is not None
    if supplied and len(list(second_seed_loglik)) != len(fits):
        raise ValueError("second_seed_loglik must have one value per fit")
    if second_seed_n_best is not None and len(list(second_seed_n_best)) != len(fits):
        raise ValueError("second_seed_n_best must have one value per fit")

    rows = []
    for i, fit in enumerate(fits):
        result = fit.result
        ic = information_criteria(fit)
        diag = classification_diagnostics(fit)
        cov = result.get("covariance") or None
        if cov == "":
            cov = None
        finite = _finite(result.get("loglik"))
        categorical = not _cov_key(cov)
        pd_flag = result.get("cov_positive_definite")
        if pd_flag is None:
            pd_flag = True if categorical else False
        collapsed = bool(result.get("cov_collapsed_component")) if result.get("cov_collapsed_component") is not None else False
        row = {
            "model_id": i + 1,
            "model": fit.model,
            "K": int(result["K"]),
            "covariance": np.nan if not _cov_key(cov) else cov,
            "n_parameters": int(result.get("n_parameters") or 0),
            "n_waves": int(result.get("n_waves") or 1),
            "n_analyzed": int(result.get("n_analyzed") or 0),
            "loglik": float(result["loglik"]) if finite else float("nan"),
            "min_class_prop": diag["min_class_prop"],
            "min_class_n": diag["min_class_n"],
            "min_avepp": diag["min_avepp"],
            "entropy_wave": result.get("entropy_wave", float("nan")),
            "entropy_sequence": result.get("entropy_sequence", float("nan")),
            "entropy_raw": ic["entropy_raw"],
            "converged": bool(result.get("converged")),
            "finite": finite,
            "regularized": bool(result.get("regularized")),
            "positive_definite": bool(pd_flag),
            "collapsed": collapsed,
            "n_starts": int(result.get("n_starts") or 0),
            "n_best": int(result.get("n_best") or 0),
            "replicate_tol": float(getattr(fit, "spec", {}).get("replicate_tol", 1e-3) or 1e-3),
            "fit": fit,
        }
        for name in INDEX_NAMES:
            row[name] = ic[name]
        rows.append(row)

    primary = _primary_name(rule, rows)
    for row in rows:
        tokens = set()
        if not row["converged"]:
            tokens.add("hard:not_converged")
        if not row["finite"]:
            tokens.add("hard:non_finite")
        if row["regularized"]:
            tokens.add("hard:regularized")
        if not row["positive_definite"]:
            tokens.add("hard:not_positive_definite")
        if row["collapsed"]:
            tokens.add("hard:collapsed")
        if row["n_starts"] < 2:
            tokens.add("hard:replication_not_assessed")
        elif row["n_best"] < 2:
            tokens.add("hard:best_loglik_not_replicated")
        if _higher_admissible_hidden(row["fit"].result):
            tokens.add("hard:higher_admissible_not_replicated")
        if not _finite(row[primary]):
            tokens.add("hard:information_criterion_undefined")
        row["preseed"] = tokens

    for i, row in enumerate(rows):
        row["second_seed_loglik"] = float("nan")
        row["second_seed_n_best"] = float("nan")
        row["second_seed_abs_gap"] = float("nan")
        row["second_seed_status"] = None
        if not rule["require_second_seed"] or not auto:
            continue
        if row["preseed"] & _PRESEED:
            continue
        if supplied:
            ll2 = float(list(second_seed_loglik)[i])
            if second_seed_n_best is None:
                row["second_seed_loglik"] = ll2
                row["second_seed_n_best"] = float("nan")
                row["second_seed_status"] = "not assessed"
                row["preseed"].add("hard:second_seed_not_assessed")
                continue
            row["second_seed_status"] = "assessed"
            nbest = int(list(second_seed_n_best)[i])
        else:
            if row["fit"].panel is None:
                ll2 = float("nan")
                nbest = 0
            else:
                refit = _refit_second_seed(row["fit"], rule["second_seed_offset"])
                ll2 = float(refit["loglik"])
                nbest = int(refit["n_best"])
            row["second_seed_status"] = "assessed"
        row["second_seed_loglik"] = ll2
        row["second_seed_n_best"] = nbest
        if _finite(ll2) and row["finite"]:
            row["second_seed_abs_gap"] = abs(ll2 - row["loglik"])
        if not _finite(ll2) or (row["finite"] and abs(ll2 - row["loglik"]) > row["replicate_tol"]):
            row["preseed"].add("hard:second_seed_not_reproduced")
        if nbest < 2:
            row["preseed"].add("hard:second_seed_not_replicated")

    for row in rows:
        tokens = set(row["preseed"])
        if _soft_fail(row["min_class_prop"], rule["min_class_prop"]):
            tokens.add("soft:min_class_prop")
        if _soft_fail(row["min_class_n"], rule["min_class_n"]):
            tokens.add("soft:min_class_n")
        if _soft_fail(row["min_avepp"], rule["min_avepp"]):
            tokens.add("soft:min_avepp")
        if rule["entropy_filter"] and _soft_fail(row["entropy_wave"], rule["min_entropy"]):
            tokens.add("soft:entropy")
        hard = [token for token in tokens if token.startswith("hard:")]
        soft = [token for token in tokens if token.startswith("soft:")]
        row["hard_pass"] = not hard
        row["soft_pass"] = not soft
        row["eligible"] = not tokens
        row["rejection_reason"] = _join(tokens)
        row["selected"] = False
        row["in_tie_zone"] = False

    _relative(rows)
    disagreement = _disagreement(rows)
    warnings = []
    wave_table = None
    joint = any(row["model"] == "lta" and int(row["n_waves"]) >= 2 for row in rows)
    if auto and waves == "auto" and joint:
        wave_table, note = _wave_summary([row["fit"] for row in rows], rule)
        if note == "configural_noninvariance_possible":
            warnings.append(note)

    blrt = {
        "ran": False, "validated": False, "switched": False, "ambiguous_kind": "",
        "reason": (
            "BLRT is off by default. The procedure follows Mplus TECH14 "
            "(p=b/B, and a draw whose best log-likelihood did not replicate is kept). "
            "Independent bootstrap p-values are not required to match. "
            "validated stays false until Monte Carlo agreement, shared-dataset likelihoods, and operating characteristics are shown. "
            "An unreplicated kept solution is a diagnostic and is not a validation failure."
        ),
        "p": float("nan"), "B": int(rule["blrt_B"]), "seconds": float("nan"),
        "starts_h0": None, "starts_h1": None, "cost_note": "",
    }
    chosen = None
    ambiguous = False
    if auto:
        chosen, ambiguous, tie = _choose(rows, primary, rule["tie_zone"])
        tie_ids = {row["model_id"] for row in tie}
        for row in rows:
            row["in_tie_zone"] = row["model_id"] in tie_ids
        blrt["ambiguous_kind"] = "exact" if ambiguous else ""
        if (
            chosen is not None and rule["blrt"] and (ambiguous or rule["blrt_check"])
            and not rule.get("experimental_blrt")
        ):
            blrt["ran"] = False
            blrt["validated"] = False
            blrt["switched"] = False
            blrt["decision"] = "not_run"
            blrt["reason"] = (
                "not run: experimental BLRT is excluded from automatic selection. "
                "blrt validated=false. Pass experimental_blrt=TRUE for a report-only result. "
                "That result does not change the selected model."
            )
        elif chosen is not None and rule["blrt"] and (ambiguous or rule["blrt_check"]):
            rival = None
            if ambiguous:
                bigger = [row for row in tie if int(row["n_parameters"]) > int(chosen["n_parameters"])]
                if bigger:
                    rival = sorted(bigger, key=lambda row: _sort_key(row, primary))[0]
            elif rule["blrt_check"]:
                others = [row for row in rows if row["eligible"] and row["model_id"] != chosen["model_id"]]
                if others:
                    rival = sorted(others, key=lambda row: _sort_key(row, primary))[0]
            if rival is None:
                blrt["reason"] = "BLRT was not run because the tie zone has no second eligible cell."
            elif chosen["fit"].panel is None or rival["fit"].panel is None:
                blrt["reason"] = "BLRT was not run because the candidate fits have no panel to simulate from."
            else:
                h0, h1 = (chosen, rival) if int(chosen["n_parameters"]) <= int(rival["n_parameters"]) else (rival, chosen)
                ran = blrt_check(
                    h0["fit"], h1["fit"], B=rule["blrt_B"], max_retries=rule["blrt_max_retries"],
                    alpha=rule["blrt_alpha"], seed=h0["fit"].spec.get("seed", 1),
                    max_iter=rule["blrt_max_iter"], experimental=True,
                )
                ran["ambiguous_kind"] = blrt["ambiguous_kind"]
                ran["switched"] = False
                ran["report_only"] = not bool(ran.get("validated"))
                if not ran.get("validated"):
                    ran["reason"] = (
                        str(ran.get("reason") or "")
                        + " Unvalidated BLRT is report-only and does not change the selected model."
                    )
                blrt = ran
        elif rule["blrt"] and not ambiguous and not rule["blrt_check"]:
            blrt["reason"] = "BLRT runs only in the tie zone or when blrt_check is TRUE."
        if chosen is not None:
            for row in rows:
                row["selected"] = row["model_id"] == chosen["model_id"]
    if auto:
        decision = "selected" if chosen is not None else "no_selection"
    else:
        decision = "auto_disabled"
    reason = _reason(decision, chosen, primary, rule, ambiguous, warnings, blrt)
    grid = "; ".join(
        f"{row['model']} K={int(row['K'])} {_cov_label(row['covariance'])}" for row in rows
    )
    reason = f"{reason} Search grid: {grid}."
    public = [{key: value for key, value in row.items() if key not in ("fit", "preseed")} for row in rows]
    comparison = compare_models(fits)
    automatic_id = None if chosen is None or not auto else int(chosen["model_id"])
    return ModelSelection(
        decision=decision,
        ambiguous=bool(ambiguous) if auto else False,
        reason=reason,
        selected_model_id=automatic_id,
        automatic_model_id=automatic_id,
        automatic_decision=decision,
        accepted=None,
        accepted_model_id=None,
        primary=primary,
        rule=rule,
        candidates=_candidate_frame(public),
        disagreement=disagreement,
        warnings=warnings,
        blrt=blrt,
        wave_enumeration=wave_table,
        fits=fits,
        comparison=comparison,
    )


def accept(selection, model_id):
    """Record the analyst's acceptance. The automatic selection is left as select_model set it."""
    if not isinstance(selection, ModelSelection):
        raise TypeError("accept() records the analyst's choice on a select_model result")
    model_id = int(model_id)
    ids = {int(value) for value in selection.candidates["model_id"]}
    if model_id not in ids:
        raise ValueError("model_id is not one of the tested candidates")
    selection.accepted_model_id = model_id
    selection.accepted = True
    return selection


def _selection_candidates(obj):
    if isinstance(obj, ModelSelection):
        return obj.candidates.copy(), obj
    if isinstance(obj, pd.DataFrame):
        return obj.copy(), None
    raise TypeError("expected a model selection or a candidate table")


def plot_ic_elbow(selection, ax=None):
    """Information-criterion elbow by covariance. Inadmissible cells are marked; the selected cell is ringed."""
    import matplotlib.pyplot as plt

    frame, _sel = _selection_candidates(selection)
    pieces = []
    for stat in ("aic", "bic", "sabic"):
        part = frame[["K", "covariance", "hard_pass", "selected", stat]].copy()
        part["criterion"] = stat.upper()
        part["value"] = part[stat]
        pieces.append(part)
    long = pd.concat(pieces, ignore_index=True)
    long = long[np.isfinite(long["value"].astype(float))]
    if long.empty:
        raise ValueError("no supported information criteria to plot")
    long["covariance"] = long["covariance"].apply(_cov_label)
    covs = list(dict.fromkeys(long["covariance"]))
    colors = {"AIC": "#1f4e79", "BIC": "#9b2226", "SABIC": "#2a9d8f"}
    fig, axes = plt.subplots(1, len(covs), squeeze=False, figsize=(4.2 * len(covs), 3.6))
    for axis, cov in zip(axes[0], covs):
        part = long[long["covariance"] == cov]
        for criterion, chunk in part.groupby("criterion", sort=False):
            chunk = chunk.sort_values("K")
            color = colors.get(criterion, "#1f4e79")
            axis.plot(chunk["K"], chunk["value"], label=criterion, color=color)
            bad = chunk[~chunk["hard_pass"].astype(bool)]
            good = chunk[chunk["hard_pass"].astype(bool)]
            if not good.empty:
                axis.scatter(good["K"], good["value"], marker="o", color=color)
            if not bad.empty:
                axis.scatter(bad["K"], bad["value"], marker="x", color="#9b2226", zorder=3)
        picked = part[part["selected"].astype(bool)]
        if not picked.empty:
            axis.scatter(picked["K"], picked["value"], s=160, facecolors="none", edgecolors="black", linewidths=1.4, zorder=4)
        axis.set_title(cov)
        axis.set_xlabel("States")
        axis.set_ylabel("Inadmissible cells are crossed and the selected cell is ringed")
        axis.legend(fontsize=8)
    fig.suptitle("Information criteria by covariance")
    if ax is not None:
        return axes[0][0]
    return axes[0][0]


def plot_avepp(fit, ax=None):
    """Heatmap of mean posteriors by modal class. The diagonal is AvePP."""
    import matplotlib.pyplot as plt

    diag = classification_diagnostics(fit)
    matrix = np.asarray(diag["avepp_matrix"], dtype=float)
    k = matrix.shape[0]
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([str(i + 1) for i in range(k)])
    ax.set_yticklabels([str(i + 1) for i in range(k)])
    ax.set_xlabel("Posterior class")
    ax.set_ylabel("Modal class")
    ax.set_title("Average posterior by modal class")
    for row in range(k):
        for col in range(k):
            value = matrix[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046)
    return ax


def plot_max_posterior(fit, ax=None):
    """Histogram of the maximum posterior probability for each person-wave."""
    import matplotlib.pyplot as plt

    result = fit.result
    cube = _gamma_cube(result)
    if cube is None:
        raise ValueError("this fit has no posterior to plot")
    post = cube.reshape(-1, cube.shape[-1])
    valid = np.isfinite(post).all(axis=1)
    values = post[valid].max(axis=1)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    ax.hist(values, bins=15, color="#1f4e79", edgecolor="white")
    ax.set_xlabel("Maximum posterior")
    ax.set_ylabel("Person-waves")
    ax.set_title("Assignment certainty")
    return ax


def plot_prevalence(fit, ax=None):
    """Expected class proportion by wave."""
    import matplotlib.pyplot as plt

    result = fit.result
    k = int(result["K"])
    waves = int(result["n_waves"])
    membership = np.asarray(result.get("expected_membership") or [], dtype=float).reshape(waves, k)
    n = float(result["n_analyzed"])
    prop = membership / n
    labels = list((fit.panel or {}).get("waves") or list(range(1, waves + 1)))
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    x = np.arange(waves)
    width = 0.8 / k
    for state in range(k):
        ax.bar(x + state * width, prop[:, state], width=width, label=f"Class {state + 1}")
    ax.set_xticks(x + width * (k - 1) / 2)
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_ylabel("Expected proportion")
    ax.set_xlabel("Wave")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Class prevalence")
    return ax


def plot_index_disagreement(selection, ax=None):
    """Which index picks which K. The table does not vote."""
    import matplotlib.pyplot as plt

    if not isinstance(selection, ModelSelection):
        raise TypeError("plot_index_disagreement expects a model selection")
    frame = selection.disagreement.dropna(subset=["K"]).copy()
    if frame.empty:
        raise ValueError("no index chose a candidate")
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    y = np.arange(len(frame))
    ax.scatter(frame["K"], y, s=60, color="#1f4e79", zorder=3)
    for i, (_, row) in enumerate(frame.iterrows()):
        ax.text(float(row["K"]) + 0.08, i, _cov_label(row["covariance"]), va="center", fontsize=8)
    if selection.decision == "selected":
        ax.axvline(selection.selected_k, color="#9b2226", linestyle="--", label="selected")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["index"].str.upper())
    ax.set_xlabel("States chosen by that index")
    ax.set_title("Index disagreement. This is not a vote.")
    ax.legend(fontsize=8)
    return ax
