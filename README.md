# pyfylr

Latent profile, latent class, and latent transition models for Python. Release 1 estimates a first-order discrete-time Markov model with a shared C++17 core (Eigen 3.4.0, no external BLAS). The same core is used by the R package profylr.

```python
import pyfylr as pf

fit = pf.fit_lta(data, indicators=["u1", "u2", "y"], id="id", wave="wave",
                 n_states=3, family={"u1": "bernoulli", "u2": "nominal", "y": "gaussian"},
                 covariance="VVV")
fit.glance()
fit.tidy()
fit.augment()
```

A cross-section is `fit_lpa` or `fit_lca`, or `data.pyfylr.fit_lpa(...)`.

The same core is available as a scikit-learn-style estimator. scikit-learn itself is optional.

```python
model = pf.LatentProfileModel(n_states=2, covariance="VVV", n_starts=40, random_state=1)
model.fit(frame[["y1", "y2"]])
model.predict(frame[["y1", "y2"]])          # modal state, 0-based
model.score(frame[["y1", "y2"]])            # mean log-likelihood, not a way to choose n_states
```

`LatentTransitionModel.predict` uses the smoothed modal state. `filter`, `smooth`, and `viterbi` are separate. `fit_result_` is the same object `fit_lpa` returns. `random_state=None` is seed 0. Do not pass `score` to `GridSearchCV` to select `n_states`.

Indicators, covariates, and distal outcomes are different roles. Release 1 estimates indicators only. A non-empty covariate specification is an error. Distal outcomes are Release 3 and require a declared timing (`initial`, `final`, `wave`, or `transition`). Modal-assignment ANOVA and logistic regression are not provided.

Supported indicator families are Gaussian, Bernoulli, and nominal. One Gaussian block may be combined with categorical indicators. The emission is the product of the block density and the categorical probabilities, so the blocks are conditionally independent given the state. Ordinal indicators are Release 2. Poisson and negative binomial indicators come after that.

The fit records `engine="cpp-em"`, an empty draws slot, `interval_type="none"`, and separates plug-in state posteriors from parameter uncertainty. No standard errors are reported in this release.

`save_model` stores an analyst selection or a comparison table. Saving a fitted model, including parameters and a way to score new rows from that file, is deferred. `predict` and `predict_proba` on the scikit-learn estimators score new rows from a model that is still in memory. The R package does not have that new-data prediction method in this release.

## Build

Release 1 supports Linux x86_64 and Apple Silicon. Windows and Intel Mac are deferred and are not supported release targets. Python 3.11, 3.12, or 3.13. NumPy is at least 2.0 on 3.11 and 3.12, and at least 2.1 on 3.13. pandas is at least 2.2.3. The build uses pybind11 at least 2.13. From the monorepo root, with a C++17 compiler and CMake:

```bash
pip install --no-build-isolation -e pyfylr
pytest pyfylr/tests
```

An editable install reads the C++ core from `../core`. The sdist vendors that core as `_staged_core` (`src/lta.cpp`, `src/describe.cpp`, `include/lta`, and `vendor/Eigen`), so `pip wheel` of the sdist does not need the monorepo beside it. Eigen's own multithreading is disabled. Random starts use private streams derived from the master seed, so one thread and several threads return the same fit.
