# contract-question-agent

![Tests](https://github.com/mofuteq/contract-question-agent/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.13-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A small open-source project that frames **contract review as an information
asymmetry problem** and generates **verification questions** that a human
reviewer can raise about a commercial contract.

The drafter of a contract typically understands its terms, edge cases, and
business implications far better than the counter-party reviewing it. The
goal of this project is to help close that gap by surfacing **concrete,
clause-grounded questions worth asking** — not by issuing verdicts.

This project is therefore explicitly:

- **Not** a contract question-answering bot.
- **Not** a legal-advice tool.
- **Not** a system that decides whether a contract is acceptable.

Any generated verification question is a **prompt for further investigation**
and **must be discussed with a qualified legal professional** before being
relied upon.

## Design rationale

Contract review usually involves an information asymmetry: the drafter
knows the clauses, edge cases, and downstream business consequences far
better than the counter-party reading them. That gap leaves reviewers
with hidden unknowns, and unknowns make signing decisions risky.

This project does **not** decide whether a clause is legal, enforceable,
or acceptable, and it does **not** decide whether anyone should sign.
Those judgments belong to a qualified legal professional and the parties
involved. Instead, the project generates **verification questions** —
concrete, clause-grounded prompts that surface what a counter-party may
want to clarify before relying on the clause.

The design draws on a few familiar ideas from economics:

- **Information asymmetry**: the reviewer has less information than the
  drafter, and targeted questions are a low-cost way to narrow that gap.
- **Screening and signaling**: questions help a reviewer screen for
  terms that warrant deeper attention and give the drafter a channel to
  signal intent.
- **Principal-agent problems**: the reviewer's interests and the
  drafter's interests are not always aligned, and explicit questions
  make that alignment testable.
- **Loss aversion**: reviewers tend to weight downside outcomes heavily,
  so surfacing unknowns up front reduces the cost of acting on them
  later.

The output is meant to **support** human and professional review, not
replace it. A generated question is a starting point for further
investigation — never a verdict.

**v0.1 shipped the data preparation layer only** — no agent, no LLM
prompts, no question generation, no clause interpretation. This repository
contains a loader for the
[Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad)
that produces JSONL files filtered to eight clause types relevant to the
project's first evaluation milestone.

**v0.2 adds a deliberately minimal end-to-end path** from CUAD
`clause_spans.jsonl` to structured `verification_questions.jsonl`. The v0.2
path uses deterministic CLI filtering, one minimal model call per clause span,
Pydantic validation, and a rule-based banned-phrase safety check. It is meant
to reveal failure patterns, not to produce high-quality legal review output.

**v0.3 updates the architecture positioning**: LangGraph owns workflow
orchestration and state transitions, MAF/OpenRouter remains the node-internal
LLM calling path for structured output, and business node logic stays
framework-independent. Optional Langfuse tracing provides a unified trace
tree, Agent Graph, business-node spans, and nested generation usage.

## Architecture and observability

```text
LangGraph:
  workflow orchestration / state transitions

FastAPI:
  thin HTTP adapter around the existing workflow

MAF + OpenRouter:
  node-internal LLM calling / structured output

Langfuse:
  unified trace tree / Agent Graph / business-node spans / generation usage

CI:
  GitHub Actions on debian:stable-slim with uv-managed Python 3.13.13
```

## Layout

```
data/cuad/
  raw/            # downloaded CUAD payload (gitignored)
  processed/      # contracts.jsonl, labels.jsonl, clause_spans.jsonl
docs/
  data.md         # data provenance, license, limitations
src/contract_question_agent/
  cuad_downloader.py  # optional downloader
  cuad_loader.py      # parser + JSONL writer
  cli_generate_questions.py
  api/                # thin FastAPI HTTP adapter
  model_client/        # MAF-backed OpenRouter LLM calling path
  workflows/
    workflow.py        # LangGraph orchestration / state transitions
    tracing.py         # optional Langfuse tracing helpers
    nodes/             # framework-independent business node transitions
tests/
  test_cuad_downloader.py
  test_cuad_loader.py
  test_generate_questions_cli.py
  test_workflow.py
  test_openrouter_client.py
  test_safety.py
  test_schemas.py
```

## Quick start

This project uses [uv](https://docs.astral.sh/uv/) and a pinned
`.python-version` to make local development reproducible with
**Python 3.13.13**. `uv sync` installs the matching interpreter (no
prior install needed), creates `.venv/`, and resolves all dependencies
from the committed `uv.lock`.

```bash
# Set up the environment (Python 3.13.13 + locked dependencies).
uv sync

# Run the test suite.
uv run pytest

# 1. Download CUAD_v1.json (Hugging Face is the default source).
uv run cuad-downloader --source huggingface
# Writes data/cuad/raw/CUAD_v1.json by default.

# 2. Process it into JSONL filtered to the v0.1 clause types.
uv run cuad-loader \
  --input data/cuad/raw/CUAD_v1.json \
  --output-dir data/cuad/processed

# 3. Run the minimal E2E generator without network access.
uv run contract-question-generate \
  --input data/cuad/processed/clause_spans.jsonl \
  --clause-type "Non-Compete" \
  --limit 3 \
  --dry-run
```

Zenodo is also supported as an alternative source:

```bash
uv run cuad-downloader --source zenodo
# Writes data/cuad/raw/CUAD_v1.zip by default. The loader reads .zip
# directly, so you can pass it straight to --input without unzipping.
```

The downloader is optional — if you already have `CUAD_v1.json` (or the
zip archive) on disk, place it under `data/cuad/raw/` and skip step 1.

See [docs/data.md](docs/data.md) for licensing and attribution requirements.

## Minimal E2E generation

The generator is intentionally a linear LangGraph workflow:

```
LOAD_CLAUSE_SPANS
-> FILTER_RECORDS
-> IS_IN_SCOPE
-> GENERATE_MINIMAL_QUESTIONS
-> REFLECT_AGAINST_SKILL_THESIS
-> SAFETY_CHECK
-> WRITE_OUTPUT
```

Filtering is deterministic and uses only CLI arguments:

```bash
uv run contract-question-generate \
  --input data/cuad/processed/clause_spans.jsonl \
  --clause-type "Non-Compete" \
  --contract-id SOME_CONTRACT_ID \
  --limit 3 \
  --offset 0 \
  --dry-run
```

For real model calls, set `OPENROUTER_API_KEY` and omit `--dry-run`.
`OPENROUTER_MODEL` is optional; `--model` overrides the environment and default.

Recommended local setup:

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY.
# .env is gitignored and must not be committed.
```

Alternative one-shell setup:

```bash
export OPENROUTER_API_KEY="..."
export OPENROUTER_MODEL="google/gemini-3-flash-preview"
```

Then run:

```bash
uv run contract-question-generate \
  --input data/cuad/processed/clause_spans.jsonl \
  --clause-type "Non-Compete" \
  --limit 3
```

Each run creates a fresh directory under `data/cuad/runs/<timestamp>/` by
default. The timestamp run id uses local time in `YYYYMMDD-HHMMSS` format.
Inside the run directory:

```
verification_questions.jsonl
run_metadata.json
run.log
```

`verification_questions.jsonl` contains the structured outputs.
`run_metadata.json` records the run settings and row-count metrics:
`rows_read`, `rows_filtered`, `rows_generated`, `safety_failed_count`, and
`rows_written`. `run.log` records the same safe lifecycle events and row counts
without logging API keys, clause text, or model output. Use `--output-dir` to
change the parent directory, `--run-id` for deterministic or manual run names,
and `--verbose` for DEBUG logs. The command fails if the run directory already
exists, so previous runs are not silently overwritten.

The default OpenRouter model is configured in
`src/contract_question_agent/model_client/openrouter.py` and can be overridden
with `OPENROUTER_MODEL` or `--model`. `OPENROUTER_API_KEY` is required for real
model calls; `OPENROUTER_MODEL` is optional. Tests use fake clients and do not
call the network.

The workflow calls `model_client.generate()` once per filtered clause span. If
`--limit 1` produces one output row but OpenRouter or provider logs show two
upstream requests, the duplicate request is likely inside the node-internal
MAF/OpenRouter structured-output path or provider-side handling, not LangGraph
workflow orchestration.

## FastAPI adapter

The API is a minimal HTTP boundary over the same workflow. It does not add
legal-advice behavior, autonomous tool calling, persistence, auth, background
jobs, WebSockets, or run history.

`POST /runs` accepts a single clause payload, writes a temporary one-row JSONL
input internally, executes the existing LangGraph workflow synchronously, and
returns workflow observability fields plus generated verification questions.

FastAPI writes local artifacts under `data/cuad/api-runs/{run_id}/`.

This adapter is intended for local development and internal poor E2E validation.
Do not expose it publicly without path restrictions, authentication, and
deployment hardening.

```bash
uv run uvicorn contract_question_agent.api.app:app --reload
```

Dry-run example:

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H "content-type: application/json" \
  -d '{
    "contract_id": "demo-contract",
    "clause_type": "Non-Compete",
    "evidence_text": "Employee will not compete for one year after termination.",
    "dry_run": true
  }'
```

## AG-UI run event stream

The AG-UI endpoint exposes a minimal human-facing event stream for observing a
single workflow run.

It is intentionally not a chat UI. The endpoint streams coarse run lifecycle
events over Server-Sent Events:

- `RUN_STARTED`
- `STEP_STARTED`
- `STEP_FINISHED`
- `STATE_SNAPSHOT`
- `RUN_FINISHED`
- `RUN_ERROR`

The stream is designed for a future run viewer. It does not add persistence,
auth, WebSockets, checkpointing, or run history. LangGraph checkpointing is
intentionally not added yet.

The event snapshot removes raw `evidence_text` from generated question outputs.
Local artifacts are still written under `data/cuad/api-runs/{run_id}/`.

```bash
curl -N -X POST http://127.0.0.1:8000/ag-ui/runs \
  -H "content-type: application/json" \
  -d '{
    "contract_id": "demo-contract",
    "clause_type": "Non-Compete",
    "evidence_text": "Employee will not compete for one year after termination.",
    "dry_run": true
  }'
```

## Streamlit AG-UI run viewer

The Streamlit viewer is a minimal human-facing observability UI.

It does not call LangGraph directly. It calls the FastAPI AG-UI SSE endpoint:

```txt
POST /ag-ui/runs
```

FastAPI remains the interface boundary, and LangGraph remains the workflow
owner. Streamlit does not directly call workflow internals, model clients, or
MCP. No checkpointing, resume support, persistence, auth, WebSockets, or run
history is added.

Start the FastAPI backend:

```bash
uv run uvicorn contract_question_agent.api.app:app --reload
```

Start the Streamlit viewer:

```bash
uv run streamlit run viewer/streamlit_app.py
```

Open the Streamlit URL shown in the terminal.

The viewer submits a single clause, reads AG-UI-compatible lifecycle events,
and renders the final safe state snapshot. It removes raw `evidence_text` from
rendered backend snapshots.

This is not a chat UI. It is a run/output viewer.

## Example output

The example below is **illustrative only**. The clause text is synthetic
and is not drawn from CUAD or any real contract. The questions are sample
verification prompts, not legal conclusions.

**Clause type**: Non-Compete

**Synthetic clause excerpt**:

> "For a period of twenty-four (24) months following termination of this
> Agreement for any reason, the Provider shall not, directly or indirectly,
> engage in any business activity that competes with the Company within
> any geographic territory in which the Company conducts business."

**Example verification questions**:

1. How is "competes with the Company" intended to be interpreted in
   practice, and is that interpretation recorded anywhere in the
   Agreement?
2. What geographic territories qualify as territories "in which the
   Company conducts business" at the time of termination, and how are
   they evidenced?
3. Does the 24-month restriction apply equally to terminations initiated
   by the Provider and to terminations initiated by the Company?
4. Are there roles, industries, or activities that are explicitly carved
   out of the restriction?
5. Is any compensation or consideration provided in exchange for the
   post-termination restriction?

**Example selected review lenses**:

```json
[
  {
    "label": "Duration",
    "source": "mcp_clause_review_hints",
    "reason": "The clause includes a 24-month post-termination period."
  },
  {
    "label": "Geographic scope",
    "source": "mcp_clause_review_hints",
    "reason": "The territory is defined by where the company conducts business."
  }
]
```

These are prompts for human review. They are not statements about whether
the clause is enforceable or reasonable, and they are not a recommendation
to sign or not sign.

## Roadmap

- **v0.1 — data layer**: done.
- **v0.2 — minimal end-to-end workflow**: done.
- **v0.3 — LangGraph orchestration + optional Langfuse tracing**: in progress.
- **v0.4 — lightweight MCP clause review hints**: next.
- **Later**: failure-driven decomposition, evaluation metrics, optional
  web search, domain skill expansion.

## Scope and disclaimers

- The processed JSONL is intended for **evaluation and research only**.
- This project does not provide legal advice and does not decide whether
  any contract is acceptable.
- Any future-generated verification questions are conversation starters
  for review by a qualified legal professional, not conclusions.
- CUAD is © The Atticus Project, distributed under CC BY 4.0.

## License

The source code in this repository is licensed under the MIT License.

CUAD data is not redistributed in this repository. CUAD is provided by The Atticus Project under the Creative Commons Attribution 4.0 International License (CC BY 4.0). Any CUAD-derived files generated locally remain subject to CUAD's attribution requirements.

This project is not affiliated with or endorsed by The Atticus Project.
