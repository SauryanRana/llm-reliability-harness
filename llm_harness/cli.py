from __future__ import annotations

import argparse
from typing import Sequence

from llm_harness.eval.runner import run_eval
from llm_harness.reporting.summarize import evaluate_gates, summarize_results, write_report_md


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm_harness", description="LLM reliability harness CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an evaluation")
    run_parser.add_argument("--dataset", default="data/tickets_eval.jsonl", help="Path to JSONL eval dataset")
    run_parser.add_argument(
        "--provider",
        default="dummy",
        choices=["dummy", "ollama", "openai"],
        help="Provider name",
    )
    run_parser.add_argument("--model", default="dummy", help="Model name")
    run_parser.add_argument("--base-url", default="http://localhost:11434", help="Provider base URL")
    run_parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")
    run_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    run_parser.add_argument(
        "--num-predict",
        "--max-tokens",
        dest="num_predict",
        type=int,
        default=320,
        help="Max generated tokens",
    )
    run_parser.add_argument("--num-ctx", type=int, default=2048, help="Ollama context window")
    run_parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Use Ollama JSON mode (format=json) with safe fallback",
    )
    run_parser.add_argument("--out-results", default="logs/results.jsonl", help="Where to write case results")
    run_parser.add_argument("--out-events", default="logs/events.jsonl", help="Where to write event logs")

    report_parser = subparsers.add_parser("report", help="Summarize results")
    report_parser.add_argument("--results", default="logs/results.jsonl", help="Path to JSONL results")
    report_parser.add_argument("--events", default="logs/events.jsonl", help="Path to JSONL event logs")
    report_parser.add_argument("--out", default="reports/report.md", help="Where to write markdown report")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            run_summary = run_eval(
                dataset_path=args.dataset,
                provider=args.provider,
                model=args.model,
                out_results_path=args.out_results,
                out_events_path=args.out_events,
                base_url=args.base_url,
                timeout_seconds=args.timeout,
                temperature=args.temperature,
                num_predict=args.num_predict,
                num_ctx=args.num_ctx,
                json_mode=args.json_mode,
            )
        except RuntimeError as exc:
            print(str(exc))
            return 2
        summary = summarize_results(args.out_results, args.out_events)
        gate_summary = evaluate_gates(summary)
        print(
            f"Completed {run_summary['total_cases']} cases. "
            f"Results: {run_summary['results_path']} "
            f"Events: {run_summary['events_path']}"
        )
        if gate_summary["passed"]:
            print("Gates: PASS")
            return 0

        print("Gates: FAIL")
        for check in gate_summary["checks"]:
            if check["passed"]:
                continue
            actual = check["actual"]
            actual_text = f"{actual:.3f}" if isinstance(actual, float) else str(actual)
            print(f"- {check['name']}: actual={actual_text}, threshold {check['threshold']}")
        return 1

    if args.command == "report":
        summary = summarize_results(args.results, args.events)
        write_report_md(summary, args.out)
        print(f"Report written to {args.out}")
        return 0

    parser.error("Unknown command")
    return 2
