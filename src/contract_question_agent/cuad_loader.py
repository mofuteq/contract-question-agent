"""CUAD dataset loader.

Reads the Contract Understanding Atticus Dataset (CUAD) JSON, filters it to a
small set of clause types relevant to ``contract-question-agent`` v0.1, and
emits three JSONL files for downstream evaluation:

* ``contracts.jsonl``     -- one record per contract.
* ``labels.jsonl``        -- one record per (contract, clause_type) pair
                             indicating whether the clause is present.
* ``clause_spans.jsonl``  -- one record per highlighted answer span.

This module is part of the v0.1 **data preparation layer only**. It does not
load LLM prompts, generate verification questions, interpret clauses, or
produce legal advice. CUAD is published by The Atticus Project under
CC BY 4.0; see ``docs/data.md`` for attribution requirements.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import IO, Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Target clause types (v0.1 evaluation scope)
# --------------------------------------------------------------------------- #

TARGET_CLAUSE_TYPES: tuple[str, ...] = (
    "Termination for Convenience",
    "Change of Control",
    "Non-Compete",
    "Exclusivity",
    "Most Favored Nation",
    "IP Ownership Assignment",
    "Indemnification",
    "Governing Law",
)


def _normalize_label(label: str) -> str:
    """Normalize a clause label so trivial spelling variants compare equal."""
    return re.sub(r"[\s_\-/]+", " ", label.strip().lower())


# Normalized lookup -> canonical label used in our outputs. Seeded with the
# target labels themselves and CUAD's known spelling variants.
CLAUSE_TYPE_LOOKUP: dict[str, str] = {
    _normalize_label(name): name for name in TARGET_CLAUSE_TYPES
}
for _alias, _canonical in (
    ("Termination For Convenience", "Termination for Convenience"),
    ("Change Of Control", "Change of Control"),
    ("Ip Ownership Assignment", "IP Ownership Assignment"),
    ("Most Favored Nation (MFN)", "Most Favored Nation"),
    ("Non Compete", "Non-Compete"),
):
    CLAUSE_TYPE_LOOKUP[_normalize_label(_alias)] = _canonical


# CUAD QA questions follow a stable template:
#   'Highlight the parts (if any) of this contract related to "<Category>" ...'
_QUESTION_CATEGORY_RE = re.compile(r'related to\s+"([^"]+)"', re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Record shapes (pydantic v2)
# --------------------------------------------------------------------------- #


class _FrozenRecord(BaseModel):
    """Base for immutable record models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ContractRecord(_FrozenRecord):
    contract_id: str
    source_file: str
    contract_text: str


class LabelRecord(_FrozenRecord):
    contract_id: str
    source_file: str
    clause_type: str
    label_present: bool


class ClauseSpanRecord(_FrozenRecord):
    contract_id: str
    source_file: str
    clause_type: str
    evidence_text: str
    start_char: int | None
    end_char: int | None
    label_present: bool


class ProcessedDataset(BaseModel):
    """Mutable container populated incrementally during parsing."""

    model_config = ConfigDict(extra="forbid")

    contracts: list[ContractRecord] = Field(default_factory=list)
    labels: list[LabelRecord] = Field(default_factory=list)
    spans: list[ClauseSpanRecord] = Field(default_factory=list)


class ProcessingStats(_FrozenRecord):
    contracts: int
    labels: int
    spans: int


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def extract_clause_type(qa: dict[str, Any]) -> str | None:
    """Return the canonical target clause type for a CUAD QA, or ``None``.

    Tries the ``id`` field first (CUAD ids are ``"<Title>__<Category>"``), then
    falls back to parsing the question text. Returns ``None`` when the QA does
    not correspond to one of :data:`TARGET_CLAUSE_TYPES`.
    """
    raw_label: str | None = None

    qid = qa.get("id")
    if isinstance(qid, str) and "__" in qid:
        raw_label = qid.rsplit("__", 1)[1]

    if raw_label is None:
        question = qa.get("question")
        if isinstance(question, str):
            match = _QUESTION_CATEGORY_RE.search(question)
            if match:
                raw_label = match.group(1)

    if raw_label is None:
        return None

    return CLAUSE_TYPE_LOOKUP.get(_normalize_label(raw_label))


def _qa_has_answer(qa: dict[str, Any]) -> bool:
    """Return True when the QA has at least one non-empty answer."""
    if qa.get("is_impossible") is True:
        return False
    answers = qa.get("answers") or []
    return any(
        isinstance(a, dict) and isinstance(a.get("text"), str) and a["text"].strip()
        for a in answers
    )


def _validate_span_offsets(
    *,
    context: str,
    text: str,
    start_char: int | None,
    end_char: int | None,
    contract_id: str,
    clause_type: str,
) -> None:
    """Warn (do not raise) if reported offsets do not slice to the evidence text."""
    if start_char is None or end_char is None:
        return
    actual = context[start_char:end_char]
    if actual != text:
        logger.warning(
            "Span offset mismatch for contract=%r clause=%r at [%d:%d]: "
            "context substring %r != evidence %r",
            contract_id,
            clause_type,
            start_char,
            end_char,
            actual,
            text,
        )


def parse_cuad(
    data: dict[str, Any],
    *,
    target_clauses: Iterable[str] = TARGET_CLAUSE_TYPES,
) -> ProcessedDataset:
    """Parse a loaded CUAD JSON document into filtered records.

    Args:
        data: The deserialised CUAD JSON (top-level dict with a ``"data"`` list).
        target_clauses: Clause types to keep. Defaults to the v0.1 scope.

    Returns:
        A :class:`ProcessedDataset` with contracts/labels/spans populated. For
        every (contract, target clause type) pair we emit exactly one label
        record; span records are emitted only for answered QAs.
    """
    targets = tuple(target_clauses)
    if not targets:
        raise ValueError("target_clauses must be non-empty")

    out = ProcessedDataset()
    seen_ids: set[str] = set()

    entries = data.get("data")
    if not isinstance(entries, list):
        raise ValueError("CUAD JSON is missing a top-level 'data' list")

    for entry in entries:
        title = entry.get("title")
        paragraphs = entry.get("paragraphs") or []
        if not isinstance(title, str) or not paragraphs:
            logger.warning("Skipping entry without title/paragraphs: %r", entry.get("title"))
            continue
        if len(paragraphs) > 1:
            # CUAD's convention is one paragraph per contract; warn but still
            # process the first to avoid silently dropping data.
            logger.warning(
                "Contract %r has %d paragraphs; using only the first.",
                title,
                len(paragraphs),
            )

        paragraph = paragraphs[0]
        context = paragraph.get("context")
        qas = paragraph.get("qas") or []
        if not isinstance(context, str):
            logger.warning("Skipping %r: paragraph context is not a string", title)
            continue

        contract_id = _disambiguate(title, seen_ids)
        seen_ids.add(contract_id)

        out.contracts.append(
            ContractRecord(
                contract_id=contract_id,
                source_file=title,
                contract_text=context,
            )
        )

        # Bucket QAs by canonical clause type so multiple QAs targeting the
        # same clause (rare but possible) collapse into a single label.
        spans_by_clause: dict[str, list[ClauseSpanRecord]] = {c: [] for c in targets}
        present_clauses: set[str] = set()

        for qa in qas:
            if not isinstance(qa, dict):
                continue
            clause_type = extract_clause_type(qa)
            if clause_type is None or clause_type not in spans_by_clause:
                continue

            answered = _qa_has_answer(qa)
            if not answered:
                continue

            present_clauses.add(clause_type)
            for answer in qa.get("answers") or []:
                if not isinstance(answer, dict):
                    continue
                text = answer.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                start = answer.get("answer_start")
                start_char = start if isinstance(start, int) and start >= 0 else None
                end_char = (
                    start_char + len(text) if start_char is not None else None
                )
                _validate_span_offsets(
                    context=context,
                    text=text,
                    start_char=start_char,
                    end_char=end_char,
                    contract_id=contract_id,
                    clause_type=clause_type,
                )
                spans_by_clause[clause_type].append(
                    ClauseSpanRecord(
                        contract_id=contract_id,
                        source_file=title,
                        clause_type=clause_type,
                        evidence_text=text,
                        start_char=start_char,
                        end_char=end_char,
                        label_present=True,
                    )
                )

        for clause_type in targets:
            present = clause_type in present_clauses
            out.labels.append(
                LabelRecord(
                    contract_id=contract_id,
                    source_file=title,
                    clause_type=clause_type,
                    label_present=present,
                )
            )
            if present:
                out.spans.extend(spans_by_clause[clause_type])

    return out


def _disambiguate(title: str, seen: set[str]) -> str:
    """Return ``title`` (or ``title#N``) ensuring uniqueness within ``seen``."""
    if title not in seen:
        return title
    counter = 2
    while f"{title}#{counter}" in seen:
        counter += 1
    return f"{title}#{counter}"


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #


def load_cuad_json(path: Path) -> dict[str, Any]:
    """Load a CUAD JSON file from disk, transparently handling ``.zip``.

    When ``path`` is a zip archive, the first ``*.json`` member is read.
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            json_members = [n for n in zf.namelist() if n.lower().endswith(".json")]
            if not json_members:
                raise ValueError(f"No JSON file found inside archive: {path}")
            with zf.open(json_members[0]) as fp:
                return _read_json(fp)
    with path.open("r", encoding="utf-8") as fp:
        return _read_json(fp)


def _read_json(fp: IO[Any]) -> dict[str, Any]:
    obj = json.load(fp)
    if not isinstance(obj, dict):
        raise ValueError("CUAD JSON root must be an object")
    return obj


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    """Write pydantic models (or plain dicts) as JSONL. Returns row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            payload = (
                record.model_dump() if isinstance(record, BaseModel) else record
            )
            fp.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            fp.write("\n")
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Top-level pipeline
# --------------------------------------------------------------------------- #


def process_cuad(
    input_path: Path,
    output_dir: Path,
    *,
    target_clauses: Iterable[str] = TARGET_CLAUSE_TYPES,
) -> ProcessingStats:
    """Load CUAD from ``input_path`` and write JSONL files into ``output_dir``."""
    data = load_cuad_json(Path(input_path))
    dataset = parse_cuad(data, target_clauses=target_clauses)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contracts_n = write_jsonl(output_dir / "contracts.jsonl", dataset.contracts)
    labels_n = write_jsonl(output_dir / "labels.jsonl", dataset.labels)
    spans_n = write_jsonl(output_dir / "clause_spans.jsonl", dataset.spans)

    return ProcessingStats(contracts=contracts_n, labels=labels_n, spans=spans_n)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cuad-loader",
        description="Filter CUAD into JSONL files for contract-question-agent v0.1.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to CUAD_v1.json (or a zip archive containing it).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write contracts.jsonl, labels.jsonl, clause_spans.jsonl.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    stats = process_cuad(args.input, args.output_dir)
    print(
        f"contracts={stats.contracts} labels={stats.labels} spans={stats.spans}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
