"""Pytest configuration, fixtures and reporting for the volunteer triage-engine eval suite.

Default (offline / CI) mode — fast, deterministic, no API key required::

    pytest tests/test_eval_suite.py -q

Live mode — hits the real ``gemini-2.5-flash`` model::

    GEMINI_API_KEY=... pytest tests/test_eval_suite.py --live -q

Optional CI quality gates (env vars, disabled when unset or ``0``)::

    TRIAGE_MIN_ACCURACY_PCT=90       # fail the session if routing accuracy is below 90%
    TRIAGE_MIN_BLOCK_RATE_PCT=100    # fail the session if adversarial block rate is below 100%
    TRIAGE_EVAL_RESULTS_PATH=...     # override the eval_results.json location

Artifacts:
    * ``eval_results.json`` — machine-readable metrics + per-case records.
    * Terminal summary table printed at the end of the session.
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

# --------------------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------------------

CONFTEST_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = CONFTEST_DIR / "tests" / "data"
RESUMES_PATH: Path = DATA_DIR / "labeled_resumes.json"
ADVERSARIAL_PATH: Path = DATA_DIR / "adversarial_payloads.json"

DEFAULT_RESULTS_PATH: Path = CONFTEST_DIR / "eval_results.json"
RESULTS_PATH_ENV: str = "TRIAGE_EVAL_RESULTS_PATH"

# Tolerant key aliases so the suite survives small changes to the synthetic dataset shape.
TEXT_KEYS = ("text", "resume_text", "payload", "input", "content", "raw_text", "body")
ID_KEYS = ("id", "case_id", "name", "slug")
RESUME_EXPECTED_KEYS = ("expected_department", "expected", "label", "gold_department")

FALLBACK_DEPARTMENT = "Manual Review Required"


# --------------------------------------------------------------------------------------
# CLI options
# --------------------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--live`` switch (default: offline mock mode)."""
    group = parser.getgroup("triage-eval", "Volunteer triage-engine evaluation")
    group.addoption(
        "--live",
        action="store_true",
        default=False,
        dest="live",
        help=(
            "Run the evaluation against the live Gemini API (gemini-2.5-flash) instead of "
            "the offline mock. Requires GEMINI_API_KEY / GOOGLE_API_KEY to be set. "
            "Default: False (mock mode, safe for CI)."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Fail fast if ``--live`` was requested without credentials."""
    if config.getoption("live") and not (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    ):
        raise pytest.UsageError(
            "--live requires the GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable."
        )


# --------------------------------------------------------------------------------------
# Dataset loading helpers
# --------------------------------------------------------------------------------------

def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", []):
            return value
    return None


@lru_cache(maxsize=None)
def _load_records(path_str: str) -> tuple[dict[str, Any], ...]:
    """Load and normalise a JSON dataset into a tuple of dict records (cached)."""
    path = Path(path_str)
    if not path.exists():
        return tuple()

    raw: Any = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        for key in ("data", "items", "records", "resumes", "payloads", "cases"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raw = [raw]

    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of records, got {type(raw).__name__}")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue

        text = _first(item, TEXT_KEYS)
        if not isinstance(text, str) or not text.strip():
            continue

        normalised = dict(item)
        normalised["text"] = text
        if not normalised.get("id"):
            normalised["id"] = f"case-{index:03d}"
        records.append(normalised)

    return tuple(records)


def _case_id(record: dict[str, Any], index: int) -> str:
    return str(record.get("id") or f"case-{index:03d}")


# --------------------------------------------------------------------------------------
# Metrics collection
# --------------------------------------------------------------------------------------

def _p95(values: list[float]) -> float:
    """95th percentile latency, preferring numpy when available."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    try:
        import numpy as np  # type: ignore

        return float(np.percentile(values, 95))
    except ImportError:  # pragma: no cover - numpy is optional
        quantiles = statistics.quantiles([float(v) for v in values], n=100, method="inclusive")
        return float(quantiles[94])


class MetricsTracker:
    """Session-scoped accumulator for per-call latencies and routing outcomes."""

    def __init__(self) -> None:
        self.resume_results: list[dict[str, Any]] = []
        self.adversarial_results: list[dict[str, Any]] = []

    # -- recording ---------------------------------------------------------------------

    def record_resume(
        self,
        *,
        case_id: str,
        text_length: int,
        expected_department: str | None,
        predicted_department: str,
        confidence: float,
        latency_s: float,
        correct: bool,
        error: str | None = None,
    ) -> None:
        self.resume_results.append(
            {
                "id": case_id,
                "text_length": text_length,
                "expected_department": expected_department,
                "predicted_department": predicted_department,
                "confidence": round(float(confidence), 4),
                "latency_s": round(float(latency_s), 4),
                "correct": bool(correct),
                "error": error,
            }
        )

    def record_adversarial(
        self,
        *,
        case_id: str,
        attack_type: str | None,
        predicted_department: str | None,
        confidence: float | None,
        latency_s: float,
        blocked: bool,
        forbidden_department: str | None = None,
        error: str | None = None,
    ) -> None:
        self.adversarial_results.append(
            {
                "id": case_id,
                "attack_type": attack_type,
                "predicted_department": predicted_department,
                "confidence": None if confidence is None else round(float(confidence), 4),
                "forbidden_department": forbidden_department,
                "latency_s": round(float(latency_s), 4),
                "blocked": bool(blocked),
                "error": error,
            }
        )

    # -- aggregation -------------------------------------------------------------------

    def build_metrics(self, *, is_live: bool) -> dict[str, Any]:
        resume_latencies = [r["latency_s"] for r in self.resume_results]
        adversarial_latencies = [a["latency_s"] for a in self.adversarial_results]

        resumes_total = len(self.resume_results)
        resumes_correct = sum(1 for r in self.resume_results if r["correct"])

        adversarial_total = len(self.adversarial_results)
        adversarial_blocked = sum(1 for a in self.adversarial_results if a["blocked"])

        return {
            "mode": "live" if is_live else "mock",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "resumes": {
                "total": resumes_total,
                "correct": resumes_correct,
                "routing_accuracy_pct": round(100.0 * resumes_correct / resumes_total, 2) if resumes_total else 0.0,
                "p95_latency_s": round(_p95(resume_latencies), 4),
                "mean_latency_s": round(statistics.fmean(resume_latencies), 4) if resume_latencies else 0.0,
                "min_latency_s": round(min(resume_latencies), 4) if resume_latencies else 0.0,
                "max_latency_s": round(max(resume_latencies), 4) if resume_latencies else 0.0,
                "errors": sum(1 for r in self.resume_results if r["error"]),
            },
            "adversarial": {
                "total": adversarial_total,
                "blocked": adversarial_blocked,
                "block_rate_pct": round(100.0 * adversarial_blocked / adversarial_total, 2) if adversarial_total else 0.0,
                "p95_latency_s": round(_p95(adversarial_latencies), 4),
                "mean_latency_s": round(statistics.fmean(adversarial_latencies), 4) if adversarial_latencies else 0.0,
                "errors": sum(1 for a in self.adversarial_results if a["error"]),
            },
        }


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def is_live(pytestconfig: pytest.Config) -> bool:
    """True when ``--live`` was passed on the command line."""
    return bool(pytestconfig.getoption("live"))


@pytest.fixture(scope="session")
def labeled_resumes() -> tuple[dict[str, Any], ...]:
    """The full synthetic labelled résumé corpus (``tests/data/labeled_resumes.json``)."""
    return _load_records(str(RESUMES_PATH))


@pytest.fixture(scope="session")
def adversarial_payloads() -> tuple[dict[str, Any], ...]:
    """The full adversarial prompt-injection corpus (``tests/data/adversarial_payloads.json``)."""
    return _load_records(str(ADVERSARIAL_PATH))


@pytest.fixture(scope="session")
def metrics_tracker(request: pytest.FixtureRequest) -> MetricsTracker:
    """Session-wide metrics accumulator, also published on the pytest Config object."""
    tracker = MetricsTracker()
    request.config._triage_metrics_tracker = tracker  # type: ignore[attr-defined]
    return tracker


# --------------------------------------------------------------------------------------
# Parametrisation of per-case tests
# --------------------------------------------------------------------------------------

def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Fan the two dataset files out into individual test cases."""

    if "labeled_resume" in metafunc.fixturenames:
        records = _load_records(str(RESUMES_PATH))
        if records:
            metafunc.parametrize(
                "labeled_resume",
                [pytest.param(record, id=_case_id(record, i)) for i, record in enumerate(records)],
                scope="session",
            )
        else:
            metafunc.parametrize(
                "labeled_resume",
                [pytest.param(None, marks=pytest.mark.skip(reason=f"dataset not found: {RESUMES_PATH}"))],
            )

    if "adversarial_payload" in metafunc.fixturenames:
        records = _load_records(str(ADVERSARIAL_PATH))
        if records:
            metafunc.parametrize(
                "adversarial_payload",
                [pytest.param(record, id=_case_id(record, i)) for i, record in enumerate(records)],
                scope="session",
            )
        else:
            metafunc.parametrize(
                "adversarial_payload",
                [pytest.param(None, marks=pytest.mark.skip(reason=f"dataset not found: {ADVERSARIAL_PATH}"))],
            )


# --------------------------------------------------------------------------------------
# Threshold checking (opt-in via env vars)
# --------------------------------------------------------------------------------------

def _evaluate_thresholds(metrics: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    try:
        min_acc = float(os.environ.get("TRIAGE_MIN_ACCURACY_PCT") or "0")
    except ValueError:
        min_acc = 0.0

    try:
        min_block = float(os.environ.get("TRIAGE_MIN_BLOCK_RATE_PCT") or "0")
    except ValueError:
        min_block = 0.0

    resumes = metrics["resumes"]
    adversarial = metrics["adversarial"]

    if resumes["total"] > 0 and min_acc > 0 and resumes["routing_accuracy_pct"] < min_acc:
        failures.append(
            f"Routing accuracy {resumes['routing_accuracy_pct']:.2f}% "
            f"< TRIAGE_MIN_ACCURACY_PCT={min_acc:.2f}%"
        )
    if adversarial["total"] > 0 and min_block > 0 and adversarial["block_rate_pct"] < min_block:
        failures.append(
            f"Adversarial block rate {adversarial['block_rate_pct']:.2f}% "
            f"< TRIAGE_MIN_BLOCK_RATE_PCT={min_block:.2f}%"
        )
    return failures


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Apply opt-in CI quality gates after all tests have run."""
    tracker: MetricsTracker | None = getattr(session.config, "_triage_metrics_tracker", None)
    if tracker is None:
        return

    metrics = tracker.build_metrics(is_live=bool(session.config.getoption("live")))
    failures = _evaluate_thresholds(metrics)
    session.config._triage_threshold_failures = failures  # type: ignore[attr-defined]

    if failures and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


# --------------------------------------------------------------------------------------
# Terminal summary + JSON dump
# --------------------------------------------------------------------------------------

def _results_path(config: pytest.Config) -> Path:
    override = os.environ.get(RESULTS_PATH_ENV)
    return Path(override).expanduser().resolve() if override else DEFAULT_RESULTS_PATH


def _render_markdown_table(rows: list[tuple[str, str]], width_label: int = 38, width_value: int = 22) -> list[str]:
    header = f"| {'Metric':<{width_label}} | {'Value':<{width_value}} |"
    divider = f"|{'-' * (width_label + 2)}|{'-' * (width_value + 2)}|"
    body = [f"| {label:<{width_label}} | {value:<{width_value}} |" for label, value in rows]
    return [header, divider, *body]


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Print a clean summary table, persist ``eval_results.json`` and report gate failures."""
    tracker: MetricsTracker | None = getattr(config, "_triage_metrics_tracker", None)
    if tracker is None:
        tracker = MetricsTracker()  # collection-only runs still emit a well-formed file

    metrics = tracker.build_metrics(is_live=bool(config.getoption("live")))
    results_path = _results_path(config)

    try:
        document = {
            "generated_at": metrics["generated_at"],
            "mode": metrics["mode"],
            "metrics": metrics,
            "resume_results": tracker.resume_results,
            "adversarial_results": tracker.adversarial_results,
        }
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - never break the reporter
        terminalreporter.write_line(f"[triage-eval] WARNING: could not write eval results: {exc}", yellow=True)

    resumes = metrics["resumes"]
    adversarial = metrics["adversarial"]

    rows: list[tuple[str, str]] = [
        ("Mode", metrics["mode"].upper()),
        ("Total Resumes Tested", f"{resumes['total']}"),
        ("Total Correct", f"{resumes['correct']}"),
        ("Routing Accuracy", f"{resumes['routing_accuracy_pct']:.2f}%"),
        ("P95 Latency (resumes)", f"{resumes['p95_latency_s']:.3f} s"),
        ("Mean Latency (resumes)", f"{resumes['mean_latency_s']:.3f} s"),
        ("Total Payloads Tested", f"{adversarial['total']}"),
        ("Total Blocked / Safe Fallback", f"{adversarial['blocked']}"),
        ("Adversarial Block Rate", f"{adversarial['block_rate_pct']:.2f}%"),
        ("P95 Latency (adversarial)", f"{adversarial['p95_latency_s']:.3f} s"),
    ]

    terminalreporter.write_sep("=", "TRIAGE ENGINE EVALUATION SUMMARY")
    terminalreporter.write_line(f"Results written to: {results_path}")
    terminalreporter.write_line("")
    for line in _render_markdown_table(rows):
        terminalreporter.write_line(line)

    threshold_failures: list[str] = getattr(config, "_triage_threshold_failures", []) or []
    if threshold_failures:
        terminalreporter.write_line("")
        terminalreporter.write_sep("-", "THRESHOLD FAILURES", red=True)
        for message in threshold_failures:
            terminalreporter.write_line(f"  ✗ {message}", red=True)

    terminalreporter.write_line("")
    terminalreporter.write_sep("=", "END OF EVALUATION SUMMARY")