"""
Definitions of nox jobs.
"""
from pathlib import Path

import nox

PYSOLARSAIL_DIR = Path(__file__).parent
TESTS_DIR = PYSOLARSAIL_DIR / "tests"
SRC_DIR = PYSOLARSAIL_DIR / "src" / "pysolarsail"
UNIT_DIR = TESTS_DIR / "unit"
VALIDATION_DIR = TESTS_DIR / "validation"
PYPROJECT_FILE = PYSOLARSAIL_DIR / "pyproject.toml"
LINT_DIRS = (str(SRC_DIR), str(TESTS_DIR))

ENV_VARS = {
    "NUMBA_DISABLE_JIT": "1",
    "PYSOLARSAIL_DO_PLOT": "0",
}

UNIT_TEST_DEPS = (
    "pytest==7.1.3",
    "pytest-cov==4.0.0",
    "parameterized==0.8.1",
)

LINT_DEPS = (
    "black==22.8",
    "flake8==3.8.4",
    "isort==5.6.4",
    "mypy==0.910",
    "numpy>=1.20",
)

nox.options.sessions = ["lint", "unit", "validation"]


def install_all(session, extra_deps):
    """Install all deps."""
    session.install("--upgrade", "pip")
    session.install("-r", "requirements.txt")
    session.install(*extra_deps)
    session.install("-e", f"{PYSOLARSAIL_DIR}")


@nox.session
def lint(session):
    """Linting."""
    install_all(session, LINT_DEPS)
    session.run(
        "isort",
        *LINT_DIRS,
        "--profile", "black",
        "--line-length", "88",
        "--diff",
        "--check",
    )
    session.run("black", *LINT_DIRS, "--diff", "--check")
    session.run("flake8", *LINT_DIRS, "--max-line-length", "88")
    #session.run("mypy")  # disabled see #4


@nox.session
def unit(session):
    """Unit tests."""
    install_all(session, UNIT_TEST_DEPS)
    session.run(
        "pytest", "-m", "unit", "-vv",
        f"{UNIT_DIR}",
        "--cov",
        "--cov-report=xml:coverage.xml",
        "--cov-report=term-missing",
        "--cov-fail-under", "50",
        env=ENV_VARS,
    )


@nox.session
def validation(session):
    """Validation tests for the solarsail model."""
    install_all(session, UNIT_TEST_DEPS)
    session.run(
        "pytest", "-vv", f"{VALIDATION_DIR}",
        env=ENV_VARS,
    )
