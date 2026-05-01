# contract-question-agent

A small open-source project that will (eventually) answer questions about
commercial contracts. **v0.1 ships the data preparation layer only** — no
agent, no LLM prompts, no legal advice.

This repository currently contains a loader for the
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
  cuad_loader.py  # parser + JSONL writer
tests/
  test_cuad_loader.py
```

## Quick start

```bash
pip install -e ".[dev]"
pytest

# After obtaining CUAD_v1.json (see docs/data.md):
python -m contract_question_agent.cuad_loader \
  --input data/cuad/raw/CUAD_v1.json \
  --output-dir data/cuad/processed
```

See [docs/data.md](docs/data.md) for licensing and attribution requirements.

## Scope and disclaimers

- The processed JSONL is intended for **evaluation and research only**.
- Nothing in this repository is, or should be relied upon as, legal advice.
- CUAD is © The Atticus Project, distributed under CC BY 4.0.
