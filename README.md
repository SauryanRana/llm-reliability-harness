# LLM Reliability Harness for IT Ticket Triage

## Project overview
This project is an LLM reliability + evaluation harness for IT support ticket triage. It tests whether an LLM can extract structured `TicketSignals` JSON from free-text tickets, then applies deterministic rules to produce final triage labels.

The workflow is: LLM extracts signals -> rules compute labels -> outputs are compared against a golden set -> quality gates decide pass/fail. This design makes model behavior measurable, repeatable, and safer to improve over time.

## Why this approach (signals -> rules) vs "LLM decides labels"
If the LLM directly decides final labels, results can be less stable and harder to debug. Small prompt/model changes can move category or priority unexpectedly.

With this harness, the LLM only extracts structured signals. Final decisions come from deterministic rules (`llm_harness/eval/rules.py`). This gives:
- Better consistency across runs
- Easier debugging (signal extraction issue vs rule issue)
- Safer regression testing with clear pass/fail gates
- Faster targeted fixes without changing schema or prompts too much

## Pipeline diagram (ASCII)
```text
+---------------------------+
| Raw IT ticket text        |
+------------+--------------+
             |
             v
+---------------------------+
| LLM provider (Ollama)     |
| -> TicketSignals JSON     |
+------------+--------------+
             |
             v
+---------------------------+
| Deterministic rules       |
| category/priority/device  |
| needs_clarification       |
| missing_fields, summary   |
+------------+--------------+
             |
             v
+---------------------------+
| Scoring vs golden set     |
| + latency/token stats     |
+------------+--------------+
             |
             v
+---------------------------+
| Quality gates + report    |
| reports/report.md         |
| logs/results.jsonl        |
| logs/events.jsonl         |
+---------------------------+
```

## What is a golden set + what's inside `data/tickets_eval.jsonl`
A golden set is a trusted reference dataset with known-correct expected outputs. It is used as the source of truth for evaluation and regression checks.

Each row in `data/tickets_eval.jsonl` contains:
- `id`: stable case ID (for regression tracking)
- `input_text`: raw user ticket
- `expected`: target output fields
  - `category`
  - `priority`
  - `device`
  - `needs_clarification`
  - `missing_fields`
  - `summary`

## Metrics explained
- `json_valid_rate`: percent of model outputs that parse as JSON.
- `schema_valid_rate`: percent of outputs that contain required schema fields.
- `category_accuracy`: percent of correct category predictions.
- `priority_accuracy`: percent of correct priority predictions.
- `device_accuracy`: percent of correct device predictions.
- `needs_clarification_accuracy`: percent of correct clarification decisions.
- `latency_p50_ms`: median response latency.
- `latency_p95_ms`: tail latency (95th percentile), useful for SLA-like checks.
- `unknown_missing_fields_rate`: percent of predicted `missing_fields` not in allowed canonical dataset keys.

The report also includes valid-json-only accuracies to separate extraction reliability from decision quality.

## Quality gates
Quality gates are automatic pass/fail checks used to block regressions.

Typical thresholds in this repo:
- `category_accuracy >= 0.85`
- `schema_valid_rate = 1.00`
- `latency_p95_ms <= threshold`
  - Local Ollama default: `<= 6000 ms`
  - Hosted providers can use stricter thresholds (for example `<= 2000 ms`)

Why they matter:
- Prevent silent quality drops
- Keep results comparable across model/prompt/rule changes
- Enforce reliability before deployment

## Quickstart
### 1) Install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e .[dev]
```

### 2) Start Ollama and pull model
```bash
ollama serve
ollama pull qwen2.5:3b
```

### 3) Run evaluation
```bash
python -m llm_harness run --dataset data/tickets_eval.jsonl --provider ollama --model qwen2.5:3b --num-predict 320
```

Optional (if supported by the provider endpoint in your setup):
```bash
python -m llm_harness run --dataset data/tickets_eval.jsonl --provider ollama --model qwen2.5:3b --num-predict 320 --json-mode
```

### 4) Generate report
```bash
python -m llm_harness report --results logs/results.jsonl --events logs/events.jsonl --out reports/report.md
```

## Outputs
- `reports/report.md`: human-readable evaluation summary and failure breakdown.
- `logs/results.jsonl`: per-case expected vs actual, correctness flags, failure reasons.
- `logs/events.jsonl`: runtime events including latency and usage metadata.

## Project structure
```text
.
|-- data/
|   `-- tickets_eval.jsonl
|-- llm_harness/
|   |-- cli.py
|   |-- providers/
|   |   |-- ollama.py
|   |   |-- openai.py
|   |   `-- dummy.py
|   |-- eval/
|   |   |-- dataset.py
|   |   |-- runner.py
|   |   |-- rules.py
|   |   |-- schema.py
|   |   |-- scoring.py
|   |   `-- normalize.py
|   |-- reporting/
|   |   `-- summarize.py
|   `-- logging/
|       `-- events.py
|-- logs/
|-- reports/
|-- tests/
`-- README.md
```

## How to interpret failures
Start with `Top Failure Causes` in `reports/report.md`:
- `InvalidJSON`: model text could not be parsed as JSON.
- `SchemaError`: JSON parsed but required fields failed validation.
- `EmptyOutput`: no model output.
- `ExtractionFailure`: extra chatter/no extractable JSON object.
- `RuleConflict`: rule-level consistency issue.

Then check failed case IDs in `## Failures`:
- Compare expected vs actual fields.
- Decide if issue is extraction (LLM) or decision logic (rules).
- Add or adjust deterministic tests for the failing pattern.

## Extending the project
- Add more golden cases in `data/tickets_eval.jsonl` for new ticket patterns.
- Compare models/providers by running the same dataset with different `--provider` / `--model`.
- Tighten quality gates as reliability improves.
- Build dashboards from `logs/results.jsonl` and `logs/events.jsonl` for trend tracking over time.

## CV bullet points
- Built an LLM reliability harness for IT ticket triage using a hybrid architecture (LLM signal extraction + deterministic rule engine), achieving stable 100% JSON/schema validity on a golden dataset.
- Designed regression-oriented evaluation with per-field accuracy, latency p50/p95, failure-cause taxonomy, and automated quality gates for CI-friendly pass/fail decisions.
- Improved production readiness by canonicalizing missing-field outputs, adding targeted rule overrides/tests, and generating actionable markdown reports from JSONL run artifacts.
