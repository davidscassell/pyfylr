# TECH14 number fixtures

These CSV files replace Mplus TECH14 output that used to sit beside them.
The output files are not in this repository. Each CSV holds five printed
tokens and nothing else. The comparison tests still use the same absolute
tolerances on these numbers.

Mplus was version 8.3 (Mac). The two former copies of each output, under
`fixtures/mplus/` and `profylr/tests/testthat/fixtures/mplus/`, were
byte-identical.

## Identities

| Case | Output sha256 | Input | Input sha256 | Data | Data sha256 |
|---|---|---|---|---|---|
| lca_bernoulli_k2 | `21e866588c7368f0e807d73db04e71e61440c6b38425012f9d21cb21df2dc064` | `lca_bernoulli_k2_tech14.inp` | `879d535590a1b523b33bc1cdcd8873217f453731b250940e38e348bdeceb6cc7` | `lca_bernoulli_k2.dat` | `f7f5774d2530f59d433d8350412b0112f93de4a506ec6db7790b855d97acbab8` |
| lpa_eee_k2 | `ac0e212cd44a8f17ccb54732c97c312ab353549bbf33eae8f845628c6becad4f` | `lpa_eee_k2_tech14.inp` | `1c3c3eacfabbb60a5cbe2387a363176ad616fb7601b57187e6a90941238c8380` | `lpa_eee_k2.dat` | `b5f7208c09a292e0b4220bd3ced8007abbd14ee11f3f3c37c43004095783b85c` |
| lpa_eei_k2 | `34d46892ce40585b39c719bb32000a579b8225b28e0a7eda68bbdf062a8947bf` | `lpa_eei_k2_tech14.inp` | `4b7309391b1665617fb1338f04ab11a69d9b53665672e84da50b2594635e3978` | `lpa_eei_k2.dat` | `4cf7ed1c8508a3e46d5cb1cee82956c838dc120505061331e7bd062701415845` |

The input files name the data files and the TECH14 settings (`LRTBOOTSTRAP = 100`, `STARTS = 20 4`, `LRTSTARTS = 20 4 20 4`, `PROCESSORS = 1`). Those inputs remain in this directory.

## Extraction

Only the numeric token was kept. No other output text was copied.

- `h0` is the number on the TECH14 line labeled `H0 Loglikelihood Value`.
- `h1` is the number on the last line labeled `H0 Value` before the TECH14 section. That line is the estimated model's loglikelihood in the fit results that precede TECH14.
- `lr` is the number on the TECH14 line labeled `2 Times the Loglikelihood Difference`.
- `p` is the number on the TECH14 line labeled `Approximate P-Value`, including the printed trailing zeros.
- `successful` is the integer at the end of the TECH14 line labeled `Successful Bootstrap Draws`.

The CSV stores those tokens as text, one header row and one data row, columns `h0,h1,lr,p,successful`.
