"""Command line interface.

agent doctor                 # what is configured, what is missing
agent run --limit 3          # answer tasks, cache results
agent score --gold gold.json # exact-match scoring of cached answers
agent submit --username me   # submit the cache
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent.config import get_settings
from agent.eval import AnswerCache, BenchmarkRunner, exact_match, score
from agent.obs.logging import configure_logging
from agent.obs.tracing import configure_tracing
from agent.tools import capability_report


def _doctor(_: argparse.Namespace) -> int:
    settings = get_settings()
    tools = capability_report(settings)
    report = {
        "provider": settings.provider or "NONE - set GROQ_API_KEY, OPENAI_API_KEY or HF_TOKEN",
        "model": settings.model or "-",
        "tracing": settings.tracing_enabled,
        "tools": tools,
        "budgets": {
            "supervisor_steps": settings.max_supervisor_steps,
            "per_question_timeout_s": settings.per_question_timeout_s,
            "total_budget_s": settings.total_budget_s,
        },
        "paths": {
            "logs": str(settings.log_dir),
            "answers": str(settings.answer_cache),
            "metrics": str(settings.metrics_file),
        },
    }
    print(json.dumps(report, indent=2))
    missing = [name for name, ok in tools.items() if not ok]
    if missing:
        print(f"\nDegraded tools (missing credentials): {', '.join(missing)}", file=sys.stderr)
    return 0 if settings.provider else 1


def _load_questions(args: argparse.Namespace, runner: BenchmarkRunner) -> list[dict[str, Any]]:
    if args.questions_file:
        items = json.loads(Path(args.questions_file).read_text(encoding="utf-8"))
    else:
        items = runner.fetch_questions()
    if args.task_id:
        items = [item for item in items if item.get("task_id") == args.task_id]
    if args.limit:
        items = items[: args.limit]
    return list(items)


def _run(args: argparse.Namespace) -> int:
    configure_tracing()
    runner = BenchmarkRunner()
    questions = _load_questions(args, runner)
    print(f"Running {len(questions)} task(s)...\n")

    for progress in runner.run(questions, reuse_cache=not args.no_cache):
        print(progress.message)
        if progress.metric and progress.metric.answer:
            print(f"    -> {progress.metric.answer[:160]}")

    summary = runner.recorder.summary()
    if args.gold:
        gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
        summary["exact_match"] = str(score(runner.cache.load(), gold))

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("errors", 0) == 0 else 1


def _score(args: argparse.Namespace) -> int:
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    predictions = AnswerCache().load()
    report = score(predictions, gold)
    print(f"exact match: {report}")

    for task_id, expected in gold.items():
        got = predictions.get(task_id, "<missing>")
        hit = task_id in predictions and exact_match(got, expected)
        print(f"  [{'PASS' if hit else 'FAIL'}] {task_id}: got {got!r} expected {expected!r}")
    return 0 if report.correct == report.graded else 1


def _submit(args: argparse.Namespace) -> int:
    runner = BenchmarkRunner()
    try:
        result = runner.submit(args.username, args.agent_code)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"Submission failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="print configuration and tool availability").set_defaults(
        func=_doctor
    )

    run = sub.add_parser("run", help="answer benchmark tasks")
    run.add_argument("--limit", type=int, default=0, help="only the first N tasks")
    run.add_argument("--task-id", help="run a single task")
    run.add_argument("--questions-file", help="local JSON instead of the scoring API")
    run.add_argument("--gold", help="JSON mapping task_id -> expected answer")
    run.add_argument("--no-cache", action="store_true", help="ignore cached answers")
    run.set_defaults(func=_run)

    scorer = sub.add_parser("score", help="score cached answers against gold")
    scorer.add_argument("--gold", required=True)
    scorer.set_defaults(func=_score)

    submit = sub.add_parser("submit", help="submit cached answers")
    submit.add_argument("--username", required=True)
    submit.add_argument("--agent-code", default="https://github.com/3MagicLabs/agent")
    submit.set_defaults(func=_submit)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
