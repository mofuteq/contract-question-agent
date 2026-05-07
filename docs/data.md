# Data layer

## What is CUAD?

The **Contract Understanding Atticus Dataset (CUAD)** is a corpus of 510
commercial contracts manually annotated by legal experts with 41 categories
of clauses ("Governing Law", "Termination For Convenience", "Change Of
Control", etc.). It was released by **The Atticus Project** in collaboration
with researchers at UC Berkeley, in support of the NeurIPS 2021 paper
*"CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review"*.

Project home: https://www.atticusprojectai.org/cuad
Source repository: https://github.com/TheAtticusProject/cuad
HuggingFace mirror: https://huggingface.co/datasets/theatticusproject/cuad-qa

The dataset is distributed as a SQuAD-style JSON file
(`CUAD_v1.json`), where every contract appears once and each of the 41
clause categories is encoded as a question whose answers are highlighted
spans in the contract text.

## License and attribution

CUAD is released under the **Creative Commons Attribution 4.0 International
license (CC BY 4.0)**. Anyone redistributing CUAD or derivative data must:

1. Provide attribution to The Atticus Project.
2. Indicate if changes were made.
3. Provide a link to the license.

When you publish anything built on the JSONL files produced by this loader,
include the following notice (or equivalent):

> Contains data from the Contract Understanding Atticus Dataset (CUAD),
> © The Atticus Project, used under CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/). The data has been filtered
> to a subset of clause types and reformatted as JSONL.

The contracts themselves are public-record SEC filings; the value-add (and
the thing that is licensed) is the expert annotation.

## Obtaining the raw data

This repository **does not vendor CUAD**. Download it yourself, either with
the bundled downloader script or by hand:

### Option 1: bundled downloader (recommended)

```bash
python -m contract_question_agent.cuad_downloader \
  --source huggingface \
  --output data/cuad/raw/CUAD_v1.json
```

Flags:

* `--source` (default `huggingface`) — `huggingface` fetches
  `CUAD_v1.json`; `zenodo` fetches the original `.zip` archive. The
  loader reads `.zip` directly (`load_cuad_json` extracts the first
  JSON member transparently), so unzipping is **optional**; do it
  manually if you prefer.
* `--output` — destination path. The default is **source-specific** so
  the file extension always matches the payload that was fetched:
  * `huggingface` → `data/cuad/raw/CUAD_v1.json`
  * `zenodo` → `data/cuad/raw/CUAD_v1.zip`

  Pass `--output PATH` to override. An explicit path always wins, even
  if its extension differs from the source's payload format.
* `--force` — overwrite an existing output file. Without `--force`, an
  existing file is preserved and the network is not contacted.
* `--verbose` — log the source URL and destination at INFO level.

The script streams the payload in 64 KiB chunks and writes through a
`*.part` file so a partial download is never confused with a complete
one.

### Option 2: download manually

* Hugging Face dataset:
  https://huggingface.co/datasets/theatticusproject/cuad
* Official GitHub release:
  https://github.com/TheAtticusProject/cuad/releases
* Zenodo archive:
  https://zenodo.org/record/4595826

Place the resulting `.json` (or unzipped contents of the archive) under
`data/cuad/raw/`, e.g.:

```
data/cuad/raw/CUAD_v1.json
```

The `data/cuad/raw/` and `data/cuad/processed/` directories are gitignored.

## Running the loader

Once `pip install -e ".[dev]"` has installed the package:

```bash
python -m contract_question_agent.cuad_loader \
  --input data/cuad/raw/CUAD_v1.json \
  --output-dir data/cuad/processed
```

You can also point `--input` at a `.zip` archive — the loader will read the
first `*.json` member.

The command writes three JSONL files into `--output-dir`:

| File | One row per | Fields |
|---|---|---|
| `contracts.jsonl` | contract | `contract_id`, `source_file`, `contract_text` |
| `labels.jsonl` | (contract, target clause type) | `contract_id`, `source_file`, `clause_type`, `label_present` |
| `clause_spans.jsonl` | highlighted answer span | `contract_id`, `source_file`, `clause_type`, `evidence_text`, `start_char`, `end_char`, `label_present` |

`labels.jsonl` is dense: every contract gets exactly one row per target
clause type, with `label_present=false` when the annotators marked the
clause absent. `clause_spans.jsonl` is sparse: it only contains rows for
clauses that *are* present, one row per highlighted answer.

## Target clause types (v0.1 scope)

The loader filters CUAD's 41 categories down to the eight relevant to the
v0.1 evaluation:

1. Termination for Convenience
2. Change of Control
3. Non-Compete
4. Exclusivity
5. Most Favored Nation
6. IP Ownership Assignment
7. Indemnification
8. Governing Law

CUAD's spelling variants ("Termination For Convenience", "Ip Ownership
Assignment", etc.) are normalized to the canonical labels above.

## Limitations of the processed data

* **Annotation scope.** CUAD annotates 41 clause categories; we keep only
  eight. Anything outside that scope is silently dropped.
* **Annotation quality.** CUAD is high-quality but not perfect. Annotators
  occasionally miss or over-highlight spans. Some clause types are highly
  imbalanced (most contracts do not contain a "Most Favored Nation"
  clause, for example), which complicates evaluation.
* **Span granularity.** `start_char` / `end_char` are derived from CUAD's
  `answer_start` plus `len(text)`. They are reliable when present, but
  some entries lack `answer_start`; in that case both fields are `null`.
* **One paragraph per contract.** CUAD's convention is exactly one
  paragraph per contract entry. If a future release breaks that, the
  loader currently uses only the first paragraph and logs a warning.
* **Source filename.** CUAD's JSON does not carry the original PDF
  filename; we use the entry `title` as `source_file`. Duplicate titles
  are disambiguated as `Title#2`, `Title#3`, ... in `contract_id`, while
  `source_file` keeps the original title.
* **Not legal advice.** The processed data is intended for benchmarking
  and research only. Outputs derived from it must not be relied upon as
  legal advice.
