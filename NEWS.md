# pyfylr

## 0.1.1

* The wrapper refuses a fit when fewer analyzed subjects than `n_states`
  have an observed indicator. The error is `ValueError`: `too few
  analyzed subjects to identify {k} states: {n} subject(s) have an
  observed indicator`. The core and the published defaults are unchanged.
* `test_module_collection_without_sklearn_skips_checks` uses the test
  file's own path, so the check runs from an extracted sdist as well as
  from a checkout.
* The README states that 0.1.0 ships an sdist and CPython 3.11–3.13
  wheels for manylinux x86_64, macOS arm64, macOS x86_64, and Windows
  amd64. The release workflow builds those wheels and smoke-imports each
  installed wheel. That smoke import is not a broader test of Windows or
  of Intel Mac.
