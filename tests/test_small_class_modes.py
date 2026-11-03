"""The 36 shared-dataset draws whose Mplus mode is a small class.

The bootstrap comparison budget (20 starts, screen of 10, 4 full starts)
must reach that mode. Package defaults stay 40/10/4. Replication is not
required. The log-likelihood must also match the C++ probe, which is the
same number the R test requires, so the two languages agree.
"""

import pandas as pd

import pyfylr as pf

from fixture_paths import fixtures_dir

ROOT = fixtures_dir() / "blrt-small-class"
CTRL = dict(
    n_starts=20,
    screen_iter=10,
    n_full=4,
    max_iter=200,
    tol=1e-7,
    replicate_tol=1e-3,
    align_labels=False,
    n_threads=1,
)


def test_small_class_modes_reach_the_mplus_likelihood():
    manifest = pd.read_csv(ROOT / "manifest.csv")
    assert len(manifest) == 36
    assert int((manifest.fixture == "lpa_eei_k2").sum()) == 30
    assert int((manifest.fixture == "lca_bernoulli_k2").sum()) == 6
    for row in manifest.itertuples(index=False):
        path = ROOT / row.source / row.fixture / f"draw_{int(row.draw):03d}.csv"
        frame = pd.read_csv(path)
        controls = dict(CTRL, seed=int(row.seed))
        label = f"{row.source} {row.fixture} {int(row.draw)}"
        if row.fixture == "lca_bernoulli_k2":
            fit = pf.fit_lca(frame, ["u1", "u2", "u3", "u4"], n_states=2, **controls)
        else:
            fit = pf.fit_lpa(
                frame, ["y1", "y2", "y3"], n_states=2, covariance="EEI", **controls
            )
        result = fit.result
        assert result["admissible"] is True, label
        assert result["converged"] is True, label
        assert result["loglik"] >= float(row.seed_ll) - 0.01, label
        assert result["loglik"] >= float(row.mplus_h1_ll) - 0.01, label
        assert abs(result["loglik"] - float(row.core_ll)) < 1e-6, (
            f"{label} ll={result['loglik']} core={row.core_ll}"
        )
