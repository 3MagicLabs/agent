"""Hugging Face Space entry point.

A thin UI shim: all logic lives in ``src/agent``. Answers are computed and
cached first, then submitted as a separate action, so a dropped connection or a
Space restart never loses a completed run.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.config import get_settings
from agent.eval import BenchmarkRunner
from agent.obs import configure_logging, configure_tracing, read_metrics
from agent.tools import capability_report

log = configure_logging()
configure_tracing()

EMPTY = pd.DataFrame()


def run_agent(reuse_cache: bool) -> Iterator[tuple[str, pd.DataFrame, str]]:
    """Answer every task, streaming progress so the connection stays alive."""
    runner = BenchmarkRunner()
    rows: list[dict[str, object]] = []

    yield "Fetching questions...", EMPTY, ""
    try:
        questions = runner.fetch_questions()
    except Exception as exc:
        log.exception("Could not fetch questions")
        yield f"Error fetching questions: {exc}", EMPTY, ""
        return

    try:
        for progress in runner.run(questions, reuse_cache=reuse_cache):
            if progress.metric is not None:
                metric = progress.metric
                rows.append(
                    {
                        "Task ID": metric.task_id,
                        "Question": metric.question[:200],
                        "Answer": metric.answer or f"[{metric.status}] {metric.error}",
                        "Status": metric.status,
                        "Latency (s)": metric.latency_s,
                    }
                )
            yield (
                progress.message,
                pd.DataFrame(rows) if rows else EMPTY,
                json.dumps(runner.recorder.summary(), indent=2),
            )
    except Exception as exc:
        log.exception("Run failed")
        yield f"Run failed: {exc}", pd.DataFrame(rows) if rows else EMPTY, ""


def submit_cached(profile: gr.OAuthProfile | None) -> str:
    """Submit whatever is in the answer cache."""
    if profile is None:
        return "Please log in to Hugging Face with the button above."

    space_id = os.getenv("SPACE_ID", "")
    agent_code = (
        f"https://huggingface.co/spaces/{space_id}/tree/main"
        if space_id
        else "https://github.com/3MagicLabs/agent"
    )
    try:
        result = BenchmarkRunner().submit(profile.username, agent_code)
    except Exception as exc:
        log.exception("Submission failed")
        return f"Submission failed: {exc}"

    return (
        f"Submission successful!\n"
        f"User: {result.get('username')}\n"
        f"Score: {result.get('score', 'N/A')}% "
        f"({result.get('correct_count', '?')}/{result.get('total_attempted', '?')} correct)\n"
        f"Message: {result.get('message', '')}"
    )


def metrics_table() -> pd.DataFrame:
    rows = read_metrics()
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"info": "No metrics recorded yet."}])


def status_report() -> str:
    settings = get_settings()
    return json.dumps(
        {
            "provider": settings.provider or "NOT CONFIGURED",
            "model": settings.model or "-",
            "tracing": settings.tracing_enabled,
            "tools": capability_report(settings),
        },
        indent=2,
    )


with gr.Blocks(title="3MagicLabs Agent") as demo:
    gr.Markdown("# 3MagicLabs Agent — benchmark runner")
    gr.Markdown(
        """
        1. Log in to Hugging Face.
        2. **Run agent** — answers stream in and are cached to disk.
        3. **Submit cached answers** — a separate, fast request, so a long run
           can never be lost to a dropped connection.
        """
    )

    gr.LoginButton()

    reuse = gr.Checkbox(value=True, label="Reuse cached answers (skip already-answered tasks)")
    with gr.Row():
        run_button = gr.Button("Run agent", variant="primary")
        submit_button = gr.Button("Submit cached answers")

    status_output = gr.Textbox(label="Status", lines=4, interactive=False)

    with gr.Tab("Answers"):
        results_table = gr.DataFrame(label="Questions and answers", wrap=True)
    with gr.Tab("Metrics"):
        summary_output = gr.Code(label="Run summary", language="json")
        refresh_metrics = gr.Button("Refresh metric history")
        history_table = gr.DataFrame(label="Per-task metrics", wrap=True)
    with gr.Tab("Configuration"):
        config_output = gr.Code(label="Resolved configuration", language="json")
        refresh_config = gr.Button("Refresh")

    run_button.click(
        fn=run_agent, inputs=[reuse], outputs=[status_output, results_table, summary_output]
    )
    submit_button.click(fn=submit_cached, outputs=[status_output])
    refresh_metrics.click(fn=metrics_table, outputs=[history_table])
    refresh_config.click(fn=status_report, outputs=[config_output])
    demo.load(fn=status_report, outputs=[config_output])


if __name__ == "__main__":
    settings = get_settings()
    log.info("provider=%s model=%s", settings.provider, settings.model)
    log.info("tools: %s", capability_report(settings))

    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.getenv(
            "GRADIO_SERVER_NAME",
            "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1",  # noqa: S104
        ),
        debug=False,
        share=False,
    )
