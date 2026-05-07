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

**v0.1 ships the data preparation layer only** — no agent, no LLM
prompts, no question generation, no clause interpretation. This repository
currently contains a loader for the
[Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad)
that produces JSONL files filtered to eight clause types relevant to the
project's first evaluation milestone.

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
tests/
  test_cuad_downloader.py
  test_cuad_loader.py
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

## Scope and disclaimers

- The processed JSONL is intended for **evaluation and research only**.
- This project does not provide legal advice and does not decide whether
  any contract is acceptable.
- Any future-generated verification questions are conversation starters
  for review by a qualified legal professional, not conclusions.
- CUAD is © The Atticus Project, distributed under CC BY 4.0.
