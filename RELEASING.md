# Releasing pyfylr

Wheels and the sdist are built by `.github/workflows/release-pypi.yml`.
The workflow runs when a GitHub release is published, and when someone
starts it by hand. A push does not run it. Publishing uses PyPI Trusted
Publishing. The workflow has no API token and no password.

## Pending publisher on PyPI

Register the pending publisher before the project exists. The first
successful publish creates the project. The pending publisher is already
being registered. Its values are:

- PyPI project name: `pyfylr`
- Owner: `davidscassell`
- Repository name: `pyfylr`
- Workflow name: `release-pypi.yml`
- Environment name: `pypi`

The GitHub environment on `davidscassell/pyfylr` must also be named `pypi`.

The wheel matrix is CPython 3.11, 3.12, and 3.13 on Linux x86_64,
macOS x86_64, macOS arm64, and Windows. PyPy and musllinux wheels are
not built.

## TestPyPI

The same workflow can upload to TestPyPI when it is started by hand and
the `testpypi` input is true. A release does not upload there. The
TestPyPI pending publisher uses the same owner, repository, and workflow
filename, with environment name `testpypi`.
