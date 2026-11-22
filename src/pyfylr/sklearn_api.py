"""scikit-learn-style estimators for the same C++ core as the pandas API.

`n_states` is fixed by the caller. :meth:`score` is the mean observed-data
log-likelihood of that fixed model. It is not a criterion for choosing the
number of states: a mixture's log-likelihood rises as ``n_states`` grows, so
``GridSearchCV`` (or any other search) must not select ``n_states`` by
``score``. ``SCORE_SELECTS_N_STATES`` is false for that reason.

``random_state=None`` is seed 0. It is not drawn from operating-system
entropy, so a refit with the same arguments is reproducible. The attribute
itself is left as ``None``.

State labels from :meth:`predict` are 0-based. The pandas API's
``modal_state`` is 1-based and refers to the same state. Probabilities and
parameter values are not shifted.

A wide DataFrame with ``n_waves > 1`` is wave-major, and the full ordered
column list is the schema. Evaluation rejects a later-wave rename or a
column swap even when the column count and the first-wave names still match.
A NumPy array with the same column count stays positional: after a named fit
its columns are read in that order and labeled with the fitted indicator
names, and after an unnamed fit names are not consulted. A long DataFrame
that names ``id_column`` and ``wave_column`` is not this wide-column schema.

scikit-learn is optional. When it is installed the estimators inherit
``BaseEstimator``. When it is not, ``get_params`` and ``set_params`` follow
the same constructor-signature convention, so :func:`clone` still works.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd


def _values(value):
    if value is None:
        return []
    return value

SCORE_SELECTS_N_STATES = False

try:
    from sklearn.base import BaseEstimator as _Base
    SKLEARN_AVAILABLE = True
except Exception:  # sklearn is optional
    SKLEARN_AVAILABLE = False

    class _Base:
        def get_params(self, deep=True):
            signature = inspect.signature(type(self).__init__)
            return {
                name: getattr(self, name)
                for name, parameter in signature.parameters.items()
                if name != "self"
            }

        def set_params(self, **params):
            valid = self.get_params(deep=False)
            for key, value in params.items():
                if key not in valid:
                    raise ValueError(
                        f"Invalid parameter {key!r} for estimator {type(self).__name__}."
                    )
                setattr(self, key, value)
            return self


def clone(estimator, safe=True):
    """Clone an estimator. Uses sklearn.clone when scikit-learn is installed."""
    if SKLEARN_AVAILABLE:
        from sklearn.base import clone as sklearn_clone
        return sklearn_clone(estimator, safe=safe)
    return estimator.__class__(**estimator.get_params(deep=True))


def _seed(random_state):
    if random_state is None:
        return 0
    if isinstance(random_state, (int, np.integer)):
        return int(random_state)
    raise ValueError("random_state must be an integer or None. None is seed 0, not OS entropy.")


def _as_array(X):
    if isinstance(X, pd.DataFrame):
        return X, list(X.columns)
    if hasattr(X, "toarray"):
        raise ValueError("sparse input is not a panel. Pass a dense array and use NaN for a missing indicator.")
    array = np.asarray(X)
    if array.dtype.kind == "c" or np.iscomplexobj(array):
        raise ValueError("Complex data not supported")
    if array.dtype == object:
        for value in array.ravel():
            if isinstance(value, (dict, list, tuple, set)):
                raise TypeError("argument must be a string or a number")
        array = array.astype(float)
    if array.ndim != 2:
        raise ValueError("X must be a 2-dimensional array or a DataFrame")
    return array, None


def _require_rows(n_rows):
    if n_rows < 1:
        raise ValueError("Found array with 0 sample(s)")


class _LatentEstimator(_Base):
    """Shared fit/predict machinery. Subclasses only choose the pandas entry point."""

    _model = "lpa"

    def _more_tags(self):
        return {"allow_nan": True, "requires_y": False, "poor_score": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.target_tags.required = False
        return tags

    def _controls(self):
        return {
            "n_starts": int(self.n_starts),
            "screen_iter": int(self.screen_iter),
            "n_full": int(self.n_full),
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
            "replicate_tol": float(self.replicate_tol),
            "n_threads": int(self.n_threads),
            "seed": _seed(self.random_state),
            "align_labels": bool(self.align_labels),
        }

    def _frame(self, X):
        """Return a long data frame, an id column, a wave column or None, and indicator names."""
        raise NotImplementedError

    def _validate_X(self, X, reset):
        values, names = _as_array(X)
        if values.ndim != 2:
            raise ValueError("X must be a 2-dimensional array or a DataFrame")
        n_rows, n_features = int(values.shape[0]), int(values.shape[1])
        if n_rows < 1:
            raise ValueError("Found array with 0 sample(s)")
        if n_features < 1:
            raise ValueError(
                f"0 feature(s) (shape={tuple(values.shape)}) while a minimum of 1 is required."
            )
        if reset:
            self.n_features_in_ = n_features
            if isinstance(X, pd.DataFrame):
                self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
            elif hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
        elif hasattr(self, "n_features_in_") and n_features != int(self.n_features_in_):
            raise ValueError(
                f"X has {n_features} features, but {type(self).__name__} is expecting "
                f"{self.n_features_in_} features as input."
            )
        return values, names

    def _require_wide_dataframe_column_order(self, columns):
        """Reject a wide DataFrame whose columns are not the fitted order.

        NumPy has no names, so this is not called for an array. Call it only
        for a wide DataFrame with ``n_waves > 1`` on evaluation, after the
        column count has already been checked.
        """
        if not hasattr(self, "feature_names_in_"):
            return
        expected = list(self.feature_names_in_)
        received = list(columns)
        if received == expected:
            return
        raise ValueError(
            "DataFrame columns are an ordered schema and do not match the fitted column order. "
            f"{type(self).__name__} was fitted with columns {expected} and received {received}. "
            "A later-wave rename or swap is a different layout. "
            "A NumPy array with the same column count stays positional."
        )

    def _wide_names_for_evaluation(self, n_waves, dataframe_names):
        """Indicator names for a wide layout on evaluation, not on fit.

        A named fit stores every column. The first wave of those names is the
        training indicator list. An array uses that list by position. A
        DataFrame has already been required to match the full order, so its
        own names are the same list. An unnamed fit has no column schema:
        both an array and a DataFrame stay positional and get no names here.
        """
        if not hasattr(self, "feature_names_in_"):
            return None
        if dataframe_names is not None:
            return list(dataframe_names)
        full = list(self.feature_names_in_)
        width = int(n_waves)
        if width < 1 or len(full) % width != 0:
            return None
        return full[: len(full) // width]

    def fit(self, X, y=None):
        from pyfylr import fit_lca, fit_lpa, fit_lta

        frame, id_column, wave_column, indicators = self._frame(X, reset=True)
        controls = self._controls()
        if self._model == "lpa":
            fit = fit_lpa(
                frame, indicators, id=id_column, n_states=int(self.n_states),
                covariance=self.covariance, family=self.family, compact=False, **controls,
            )
        elif self._model == "lca":
            fit = fit_lca(
                frame, indicators, id=id_column, wave=wave_column,
                n_states=int(self.n_states), family=self.family, compact=False, **controls,
            )
        else:
            fit = fit_lta(
                frame, indicators, id=id_column, wave=wave_column,
                n_states=int(self.n_states), covariance=self.covariance,
                family=self.family, transitions=self.transitions,
                measurement_invariant=self.measurement_invariant,
                means_invariant=self.means_invariant,
                covariance_invariant=self.covariance_invariant,
                compact=False, **controls,
            )
        self.fit_result_ = fit
        self._assign_fitted(fit)
        return self

    def _assign_fitted(self, fit):
        result = fit.result
        params = result["params"]
        k = int(result["K"])
        self.weights_ = np.asarray(params["delta"], dtype=float)
        self.loglik_ = float(result["loglik"])
        self.n_iter_ = int(result["n_iter"])
        self.converged_ = bool(result["converged"])
        self.diagnostics_ = {
            name: result.get(name)
            for name in (
                "status", "status_reason", "eligible_model_comparison", "eligible_assignment",
                "eligible_measurement", "eligible_transitions", "eligible_further_starts",
                "eligibility_detail", "min_eigenvalue", "regularized", "n_boundary",
                "cov_positive_definite", "cov_ill_conditioned", "cov_collapsed_component",
                "condition_number", "min_relative_variance", "n_numerically_empty",
                "n_empty_state_waves", "n_negligible_transition_rows", "ic_supported",
                "label_order", "n_best", "n_starts", "admissible",
                "rejected_loglik", "rejected_n", "rejected_level", "rejected_reason",
            )
        }
        gaussian = int(result["n_gaussian"])
        means = np.asarray(params["means"], dtype=float) if len(params["means"]) else np.zeros((0,))
        if gaussian and means.size:
            blocks = means.size // (k * gaussian)
            shaped = means.reshape(blocks, k, gaussian)
            self.means_ = shaped[0] if blocks == 1 else shaped
        else:
            self.means_ = np.zeros((k, 0))
        self.covariances_ = _unpack_covariance(result)
        tau = np.asarray(params["tau"], dtype=float) if len(params["tau"]) else np.zeros((0,))
        if tau.size:
            blocks = tau.size // (k * k)
            shaped = tau.reshape(blocks, k, k)
            self.transition_matrix_ = shaped[0] if blocks == 1 else shaped
        else:
            self.transition_matrix_ = None

    def _evaluated(self, X):
        from pyfylr import _core

        frame, id_column, wave_column, indicators = self._frame(X, reset=False)
        from pyfylr import prepare_panel

        panel = prepare_panel(
            frame, indicators, id_column, wave_column, family=self.family,
            schema=self.fit_result_.panel,
        )
        spec = dict(self.fit_result_.spec)
        spec["align_labels"] = False
        spec["has_start"] = False
        spec["n_starts"] = 1
        raw = dict(_core.evaluate(panel, spec, self.fit_result_.result["params"]))
        from pyfylr import match_result_form

        return panel, match_result_form(raw, self.fit_result_.result)

    def _states(self, X, field):
        _panel, raw = self._evaluated(X)
        waves = int(raw["n_waves"])
        labels = np.asarray(raw[field], dtype=int).reshape(-1, waves)
        if waves == 1:
            return labels[:, 0]
        return labels

    def predict(self, X):
        """Modal state from the smoothed posterior. Labels are 0-based."""
        self._check_fitted()
        return self._states(X, "modal")

    def predict_proba(self, X):
        """Smoothed posteriors. Shape (n_subjects, K) or (n_subjects, n_waves, K)."""
        self._check_fitted()
        _panel, raw = self._evaluated(X)
        k = int(raw["K"])
        waves = int(raw["n_waves"])
        gamma = np.asarray(raw["gamma"], dtype=float).reshape(-1, waves, k)
        if waves == 1:
            return gamma[:, 0, :]
        return gamma

    def filter(self, X):
        """Modal state from the filtered probabilities. Not the default of predict."""
        self._check_fitted()
        _panel, raw = self._evaluated(X)
        k = int(raw["K"])
        waves = int(raw["n_waves"])
        filtered = np.asarray(raw["filtered"], dtype=float).reshape(-1, waves, k)
        labels = np.argmax(np.where(np.isfinite(filtered), filtered, -np.inf), axis=-1)
        missing = ~np.isfinite(filtered).any(axis=-1)
        labels = labels.astype(int)
        labels[missing] = -1
        if waves == 1:
            return labels[:, 0]
        return labels

    def smooth(self, X):
        """Modal state from the smoothed posterior. Same values as predict."""
        return self.predict(X)

    def viterbi(self, X):
        """Most probable state path. Separate from the smoothed modal state."""
        self._check_fitted()
        return self._states(X, "viterbi")

    def score_samples(self, X):
        """Per-subject observed-data log-likelihood. Excluded subjects are NaN.

        A wide array has one subject per row. A long data frame is aggregated
        to subjects, in the order of their ids, not one value per long row.
        """
        from pyfylr import _core, prepare_panel

        self._check_fitted()
        frame, id_column, wave_column, indicators = self._frame(X, reset=False)
        panel = prepare_panel(
            frame, indicators, id_column, wave_column, family=self.family,
            schema=self.fit_result_.panel,
        )
        spec = dict(self.fit_result_.spec)
        spec["align_labels"] = False
        spec["has_start"] = False
        return np.asarray(
            _core.subject_loglik(panel, spec, self.fit_result_.result["params"]), dtype=float
        )

    def score(self, X, y=None):
        """Mean log-likelihood per analyzed subject.

        Not a criterion for choosing ``n_states``. Do not pass this score to
        ``GridSearchCV`` to select the number of states. ``n_states`` stays
        the value stored on the estimator.
        """
        if SCORE_SELECTS_N_STATES:
            raise RuntimeError("score must not be used to choose n_states")
        samples = self.score_samples(X)
        # NaN marks an excluded subject. -inf is an impossible analyzed
        # likelihood and stays in the mean, so the score is -inf.
        analyzed = samples[~np.isnan(samples)]
        if analyzed.size == 0:
            return float("nan")
        return float(np.mean(analyzed))

    def aic(self, X):
        """AIC of the fitted parameter count on ``X``. ``n`` is the number of analyzed subjects."""
        return self._information(X, bic=False)

    def bic(self, X):
        """BIC of the fitted parameter count on ``X``. ``n`` is the number of analyzed subjects."""
        return self._information(X, bic=True)

    def _information(self, X, bic):
        samples = self.score_samples(X)
        analyzed = samples[~np.isnan(samples)]
        n = int(analyzed.size)
        if n < 1:
            return float("nan")
        loglik = float(np.sum(analyzed))
        p = int(self.fit_result_.result["n_parameters"])
        if bic:
            return -2.0 * loglik + p * float(np.log(n))
        return -2.0 * loglik + 2.0 * p

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)

    def _check_fitted(self):
        if not hasattr(self, "fit_result_"):
            try:
                from sklearn.exceptions import NotFittedError
            except Exception:
                NotFittedError = AttributeError
            raise NotFittedError(
                "This estimator is not fitted yet. Call fit before predict, score, or the information criteria."
            )


def _unpack_covariance(result):
    name = result.get("covariance") or ""
    cov = np.asarray(_values(result["params"]["cov"]), dtype=float)
    k = int(result["K"])
    j = int(result["n_gaussian"])
    if j == 0 or cov.size == 0 or not name:
        return None
    blocks = int(result["n_cov_blocks"])

    def full(packed):
        matrix = np.zeros((j, j))
        index = 0
        for row in range(j):
            for col in range(row + 1):
                matrix[row, col] = matrix[col, row] = packed[index]
                index += 1
        return matrix

    if name == "EEI":
        shaped = cov.reshape(blocks, j)
        return shaped[0] if blocks == 1 else shaped
    if name == "VVI":
        shaped = cov.reshape(blocks, k, j)
        return shaped[0] if blocks == 1 else shaped
    packed = j * (j + 1) // 2
    if name == "EEE":
        shaped = np.stack([full(cov[b * packed:(b + 1) * packed]) for b in range(blocks)])
        return shaped[0] if blocks == 1 else shaped
    if name == "VVV":
        out = np.zeros((blocks, k, j, j))
        for b in range(blocks):
            for state in range(k):
                start = (b * k + state) * packed
                out[b, state] = full(cov[start:start + packed])
        return out[0] if blocks == 1 else out
    return cov


def _wide_to_long(array, n_waves, names):
    n, p = array.shape
    if p % n_waves != 0:
        raise ValueError(
            f"a wide array of {p} columns is not divisible by n_waves={n_waves}. "
            "Columns are wave-major: wave 1 indicators, then wave 2, and so on."
        )
    j = p // n_waves
    if names is None:
        names = [f"x{i}" for i in range(j)]
    elif len(names) == p:
        names = list(names[:j])
    rows = []
    for i in range(n):
        for t in range(n_waves):
            rec = {"_id": i, "_wave": t + 1}
            for col in range(j):
                rec[names[col]] = array[i, t * j + col]
            rows.append(rec)
    return pd.DataFrame(rows), "_id", "_wave", names


class LatentProfileModel(_LatentEstimator):
    """One-wave Gaussian latent profile model.

    ``n_states`` is fixed by the caller. :meth:`score` is not a criterion
    for choosing it. Constructor arguments are stored unchanged. Fitted
    attributes end in an underscore and are created by :meth:`fit`.
    """

    _model = "lpa"

    def __init__(self, n_states=2, covariance="VVV", n_starts=40, random_state=None,
                 screen_iter=10, n_full=4, max_iter=200, tol=1e-7, replicate_tol=0.1,
                 n_threads=1, align_labels=True, family=None):
        self.n_states = n_states
        self.covariance = covariance
        self.n_starts = n_starts
        self.random_state = random_state
        self.screen_iter = screen_iter
        self.n_full = n_full
        self.max_iter = max_iter
        self.tol = tol
        self.replicate_tol = replicate_tol
        self.n_threads = n_threads
        self.align_labels = align_labels
        self.family = family

    def _frame(self, X, reset=False):
        values, names = self._validate_X(X, reset=reset)
        if isinstance(values, pd.DataFrame):
            frame = values.copy()
            indicators = list(frame.columns)
            frame = frame.assign(_id=np.arange(len(frame)))
            return frame, "_id", None, indicators
        frame = pd.DataFrame(values, columns=[f"x{i}" for i in range(values.shape[1])])
        frame = frame.assign(_id=np.arange(len(frame)))
        return frame, "_id", None, [f"x{i}" for i in range(values.shape[1])]


class LatentClassModel(_LatentEstimator):
    """One-wave categorical latent class model.

    ``n_waves > 1`` is a wide DataFrame or array, wave-major. On evaluation,
    a wide DataFrame must repeat the fitted column order, including later-wave
    names. A NumPy array stays positional and, after a named fit, is labeled
    with the fitted indicator names in that order.

    ``check_estimator`` does not apply: the checker supplies continuous
    arrays, and a class model needs categorical codes. Compare it with the
    pandas API on coded indicators instead.
    """

    _model = "lca"

    def __init__(self, n_states=2, n_starts=40, random_state=None, screen_iter=10, n_full=4,
                 max_iter=200, tol=1e-7, replicate_tol=0.1, n_threads=1, align_labels=True,
                 family=None, n_waves=1):
        self.n_states = n_states
        self.n_starts = n_starts
        self.random_state = random_state
        self.screen_iter = screen_iter
        self.n_full = n_full
        self.max_iter = max_iter
        self.tol = tol
        self.replicate_tol = replicate_tol
        self.n_threads = n_threads
        self.align_labels = align_labels
        self.family = family
        self.n_waves = n_waves

    def _frame(self, X, reset=False):
        values, names = self._validate_X(X, reset=reset)
        waves = int(self.n_waves)
        if isinstance(values, pd.DataFrame) and waves == 1:
            frame = values.copy()
            indicators = list(frame.columns)
            frame = frame.assign(_id=np.arange(len(frame)))
            return frame, "_id", None, indicators
        array = values.to_numpy(dtype=float) if isinstance(values, pd.DataFrame) else np.asarray(values, dtype=float)
        name_row = names if isinstance(values, pd.DataFrame) else None
        if waves > 1 and not reset:
            if isinstance(values, pd.DataFrame):
                self._require_wide_dataframe_column_order(values.columns)
            name_row = self._wide_names_for_evaluation(waves, name_row)
        return _wide_to_long(array, waves, name_row)


class LatentTransitionModel(_LatentEstimator):
    """First-order latent transition model.

    A NumPy array is wide and wave-major: ``n_waves * n_indicators`` columns.
    It stays positional. After a named wide fit, those positions are labeled
    with the fitted indicator names. A wide DataFrame uses that same layout,
    and evaluation requires the full fitted column order, so a later-wave
    rename or swap is an error. A long DataFrame is used when ``id_column``
    and ``wave_column`` are set; that path is not the wide-column schema.

    :meth:`predict` uses the smoothed modal state. :meth:`filter`,
    :meth:`smooth`, and :meth:`viterbi` stay separate.

    ``check_estimator`` is run at ``n_waves=1`` because the checker cannot
    supply a wave index. Multi-wave behaviour is tested against the pandas API.
    """

    _model = "lta"

    def __init__(self, n_states=2, covariance="VVV", n_starts=40, random_state=None,
                 n_waves=1, transitions="equal", measurement_invariant=True,
                 means_invariant=True, covariance_invariant=True, screen_iter=10, n_full=4,
                 max_iter=200, tol=1e-7, replicate_tol=0.1, n_threads=1, align_labels=True,
                 family=None, id_column=None, wave_column=None):
        self.n_states = n_states
        self.covariance = covariance
        self.n_starts = n_starts
        self.random_state = random_state
        self.n_waves = n_waves
        self.transitions = transitions
        self.measurement_invariant = measurement_invariant
        self.means_invariant = means_invariant
        self.covariance_invariant = covariance_invariant
        self.screen_iter = screen_iter
        self.n_full = n_full
        self.max_iter = max_iter
        self.tol = tol
        self.replicate_tol = replicate_tol
        self.n_threads = n_threads
        self.align_labels = align_labels
        self.family = family
        self.id_column = id_column
        self.wave_column = wave_column

    def _frame(self, X, reset=False):
        values, names = self._validate_X(X, reset=reset)
        if isinstance(values, pd.DataFrame) and self.id_column is not None and self.wave_column is not None:
            indicators = [c for c in values.columns if c not in (self.id_column, self.wave_column)]
            return values.copy(), self.id_column, self.wave_column, indicators
        waves = int(self.n_waves)
        if isinstance(values, pd.DataFrame):
            array = values.to_numpy(dtype=float)
            name_row = list(values.columns) if waves == 1 else None
            if waves == 1:
                frame = values.copy()
                indicators = list(frame.columns)
                frame = frame.assign(_id=np.arange(len(frame)))
                return frame, "_id", None, indicators
            name_row = list(values.columns)
            if not reset:
                self._require_wide_dataframe_column_order(values.columns)
                name_row = self._wide_names_for_evaluation(waves, name_row)
            return _wide_to_long(array, waves, name_row)
        if waves == 1:
            frame = pd.DataFrame(values, columns=[f"x{i}" for i in range(values.shape[1])])
            frame = frame.assign(_id=np.arange(len(frame)))
            return frame, "_id", None, [f"x{i}" for i in range(values.shape[1])]
        name_row = None
        if not reset:
            name_row = self._wide_names_for_evaluation(waves, None)
        return _wide_to_long(np.asarray(values, dtype=float), waves, name_row)


class StateCountSearch:
    """Choose ``n_states`` and covariance with ``select_model``.

    This is not a ``GridSearchCV`` and it has no ``score``. The log-likelihood
    score rises with the number of states, so it is not the selection criterion.
    """

    def __init__(self, estimator, n_states=(1, 2, 3), covariance=None, rule=None, auto=True):
        self.estimator = estimator
        self.n_states = tuple(int(k) for k in n_states)
        self.covariance = covariance
        self.rule = rule
        self.auto = bool(auto)

    def fit(self, X, y=None):
        from pyfylr import select_model

        params = self.estimator.get_params()
        if self.covariance is None:
            covs = (params.get("covariance"),) if "covariance" in params else (None,)
        elif isinstance(self.covariance, str):
            covs = (self.covariance,)
        else:
            covs = tuple(self.covariance)
        estimators = []
        fits = []
        for k in self.n_states:
            for cov in covs:
                est = clone(self.estimator)
                est.set_params(n_states=int(k))
                if cov is not None and "covariance" in est.get_params():
                    est.set_params(covariance=cov)
                est.fit(X, y)
                estimators.append(est)
                fits.append(est.fit_result_)
        self.estimators_ = estimators
        self.selection_ = select_model(fits, rule=self.rule, auto=self.auto)
        if self.selection_.decision == "selected":
            self.best_estimator_ = estimators[int(self.selection_.selected_model_id) - 1]
        else:
            self.best_estimator_ = None
        return self

    def accept(self, model_id):
        """Record the analyst's acceptance. ``best_estimator_`` stays the automatic selection."""
        from pyfylr.selection import accept as accept_selection

        self.selection_ = accept_selection(self.selection_, model_id)
        return self

    def predict(self, X):
        if self.best_estimator_ is None:
            raise ValueError("no_selection")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        if self.best_estimator_ is None:
            raise ValueError("no_selection")
        return self.best_estimator_.predict_proba(X)
