"""Shared pytest configuration for the glsolver test-suite (docs/verification.md).

* Hypothesis profiles ``ci`` (60 examples), ``dev`` (200, the default) and
  ``thorough`` (2000).  Select one with ``--hypothesis-profile NAME`` (option
  provided by Hypothesis' own pytest plugin) or the environment variable
  ``HYPOTHESIS_PROFILE``.  All profiles disable the per-example deadline: the
  solvers' running time varies too much between machines for a deadline to
  be a meaningful assertion.
* ``--runslow`` enables tests marked ``@pytest.mark.slow`` (they are skipped
  otherwise; ``-m "not slow"`` keeps working as before).
* session fixture ``solvers``: the backend registry
  (:func:`glsolver.testing.registry.available_solvers`).
"""
from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck, settings

from glsolver.testing.registry import available_solvers

_SUPPRESSED = [
    HealthCheck.too_slow,
    HealthCheck.filter_too_much,
    HealthCheck.data_too_large,
    HealthCheck.large_base_example,
]

settings.register_profile(
    "ci", max_examples=60, deadline=None, suppress_health_check=_SUPPRESSED, print_blob=True
)
settings.register_profile(
    "dev", max_examples=200, deadline=None, suppress_health_check=_SUPPRESSED, print_blob=True
)
settings.register_profile(
    "thorough",
    max_examples=2000,
    deadline=None,
    suppress_health_check=_SUPPRESSED,
    print_blob=True,
)
# The Hypothesis pytest plugin loads ``--hypothesis-profile`` afterwards (in its
# ``pytest_configure``), so the command line wins over the environment.
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow (skipped by default)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test: pass --runslow to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def solvers():
    """Name -> callable for every backend usable in this environment (oracles included)."""
    return available_solvers(include_oracles=True)
