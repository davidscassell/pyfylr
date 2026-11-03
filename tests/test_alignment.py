"""Shared numeric fixture for align_states and add_starts."""

import pandas as pd

import pyfylr as pf

from fixture_paths import fixtures_dir

FIXTURE = fixtures_dir() / "parity" / "shared.txt"


def load_fixture():
    out = {}
    for line in FIXTURE.read_text().splitlines():
        key, rest = line.split(" ", 1)
        out[key] = rest
    return out


def panel():
    rows = []
    for i in range(40):
        bit = 0 if i < 20 else 1
        rows.append({"id": i + 1, "y1": bit, "y2": bit})
    return pd.DataFrame(rows)


def reference():
    return {
        "delta": [0.2, 0.8],
        "tau": [],
        "rho": [0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1],
        "means": [],
        "cov": [],
    }


def test_align_states_and_add_starts_match_the_shared_fixture():
    locked = load_fixture()
    fit = pf.fit_lca(
        panel(), ["y1", "y2"], id="id", n_states=2, n_starts=3, n_full=2,
        screen_iter=5, max_iter=50, tol=1e-8, seed=5, replicate_tol=1e-6,
        align_labels=True,
    )
    assert fit.result["n_starts"] == int(locked["n_starts"])
    assert fit.result["n_parameters"] == int(locked["n_parameters"])
    assert abs(fit.result["loglik"] - float(locked["loglik"])) < 1e-8
    aligned, info = pf.align_states(fit, reference())
    permutation = " ".join(str(int(v)) for v in info["permutation"])
    assert permutation == locked["permutation"]
    assert int(bool(info["ambiguous"])) == int(locked["ambiguous"])
    assert abs(float(info["cost"]) - float(locked["cost"])) < 1e-8
    rho = aligned.result["params"]["rho"]
    got = [rho[i] for i in range(1, len(rho), 2)]
    expected = [float(x) for x in locked["aligned_p1"].split()]
    assert len(got) == len(expected)
    assert max(abs(a - b) for a, b in zip(got, expected)) < 1e-8
    more = pf.add_starts(fit, n_new=1, seed=2)
    assert more.result["n_starts"] == int(locked["add_n_starts"])
    assert more.result["n_best"] == int(locked["add_n_best"])
    assert abs(more.result["loglik"] - float(locked["add_loglik"])) < 1e-8
    assert more.result["loglik"] + 1e-8 >= fit.result["loglik"]
