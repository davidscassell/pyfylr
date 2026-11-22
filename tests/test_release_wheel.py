"""Fast check of an installed wheel. No fixture files."""

import inspect
import os

import numpy as np
import pandas as pd

import pyfylr as pf


def test_installed_wheel_records_the_sdist_flags():
    if os.environ.get("PYFYLR_REQUIRE_INSTALLED") == "1":
        loaded = inspect.getfile(pf).replace("\\", "/")
        assert "site-packages" in loaded, loaded
    info = pf.build_info()
    flags = info["flags"]
    assert info["engine"] == "cpp-em"
    assert "fast-math=off" in flags
    assert "ffast-math" not in flags
    compiler = info["compiler"]
    if compiler.startswith("MSVC"):
        assert "/fp:fast" not in flags
        assert "/O2" in flags
    else:
        assert "-ffp-contract=off" in flags
        assert "-fno-fast-math" in flags
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(-1.5, 0.3, 12), rng.normal(1.5, 0.3, 12)])
    data = pd.DataFrame({"y": y})
    fit = pf.fit_lpa(
        data, ["y"], n_states=2, covariance="EEI",
        n_starts=2, screen_iter=0, n_full=1, max_iter=12,
        seed=1, n_threads=1, align_labels=False,
    )
    assert fit.result["loglik"] < 0
