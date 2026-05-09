# contract-question-agent

A small open-source project that frames **contract review as an information
asymmetry problem** and (eventually) generates **verification questions**
that a human reviewer can raise about a commercial contract.

The drafter of a contract typically understands its terms, edge cases, and
business implications far better than the counter-party reviewing it. The
goal of this project is to help close that gap by surfacing **concrete,
clause-grounded questions worth asking** — not by issuing verdicts.

This project is therefore explicitly:

- **Not** a contract question-answering bot.
- **Not** a legal-advice tool.
- **Not** a system that decides whether a contract is acceptable.

Any verification question that future versions generate is a **prompt for
further investigation** and **must be discussed with a qualified legal
professional** before being relied upon.

**v0.1 shipped the data preparation layer only** — no agent, no LLM
prompts, no question generation, no clause interpretation. This repository
contains a loader for the
[Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad)
that produces JSONL files filtered to eight clause types relevant to the
project's first evaluation milestone.

**v0.2 adds a deliberately poor end-to-end path** from CUAD
`clause_spans.jsonl` to structured `verification_questions.jsonl`. It uses a
linear Microsoft Agent Framework workflow, a Microsoft Agent Framework
OpenAI-compatible Agent for OpenRouter generation, deterministic CLI
filtering, one minimal model call per clause span, Pydantic validation, and a
rule-based banned-phrase safety check. It is meant to reveal failure patterns,
not to produce high-quality legal review output.

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
  workflows/
  model_client/
tests/
  test_cuad_downloader.py
  test_cuad_loader.py
  test_generate_questions_cli.py
  test_microsoft_linear_workflow.py
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

# 3. Run the v0.2 poor E2E generator without network access.
uv run contract-question-generate \
  --input data/cuad/processed/clause_spans.jsonl \
  --output data/cuad/processed/verification_questions.jsonl \
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

## v0.2 poor E2E generation

The generator is intentionally a linear workflow:

```
LOAD_CLAUSE_SPANS
-> FILTER_RECORDS
-> GENERATE_MINIMAL_QUESTIONS
-> SAFETY_CHECK
-> WRITE_OUTPUT
-> DONE
```

Filtering is deterministic and uses only CLI arguments:

```bash
uv run contract-question-generate \
  --input data/cuad/processed/clause_spans.jsonl \
  --output data/cuad/processed/verification_questions.jsonl \
  --clause-type "Non-Compete" \
  --contract-id SOME_CONTRACT_ID \
  --limit 3 \
  --offset 0 \
  --dry-run
```

For real model calls, set `OPENROUTER_API_KEY` and omit `--dry-run`.
You may also set `OPENROUTER_MODEL`, or pass `--model` to override both the
environment variable and the default:

```bash
export OPENROUTER_API_KEY="..."
export OPENROUTER_MODEL="google/gemini-2.5-pro"

uv run contract-question-generate \
  --input data/cuad/processed/clause_spans.jsonl \
  --output data/cuad/processed/verification_questions.jsonl \
  --clause-type "Non-Compete" \
  --limit 3
```

The default OpenRouter model is configured in
`src/contract_question_agent/model_client/openrouter.py` and can be overridden
with `OPENROUTER_MODEL` or `--model`. Tests use fake clients and do not call
the network.

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
