"""Tests for ``contract_question_agent.cuad_loader``.

The tests build small in-memory CUAD-shaped fixtures rather than touching the
real dataset, so they run offline and stay fast.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from contract_question_agent.cuad_loader import (
    CLAUSE_TYPE_LOOKUP,
    TARGET_CLAUSE_TYPES,
    ClauseSpanRecord,
    ContractRecord,
    LabelRecord,
    extract_clause_type,
    load_cuad_json,
    parse_cuad,
    process_cuad,
    write_jsonl,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


CONTRACT_TEXT_A = (
    "MASTER SERVICES AGREEMENT. "  # 30 chars
    "This Agreement is governed by the laws of Delaware. "  # +52 = 82
    "Either party may terminate this Agreement for convenience on 30 days notice."
)


def _qa(qid: str, category: str, answers: list[dict]) -> dict:
    return {
        "id": qid,
        "question": (
            f'Highlight the parts (if any) of this contract related to "{category}" '
            "that should be reviewed by a lawyer."
        ),
        "answers": answers,
        "is_impossible": not answers,
    }


def _sample_cuad_doc() -> dict:
    """A two-contract CUAD document covering several edge cases."""
    answer_governing = "governed by the laws of Delaware"
    start_governing = CONTRACT_TEXT_A.index(answer_governing)
    answer_term = "terminate this Agreement for convenience"
    start_term = CONTRACT_TEXT_A.index(answer_term)

    contract_a = {
        "title": "MASTER_SERVICES_AGREEMENT",
        "paragraphs": [
            {
                "context": CONTRACT_TEXT_A,
                "qas": [
                    _qa(
                        "MASTER_SERVICES_AGREEMENT__Governing Law",
                        "Governing Law",
                        [{"text": answer_governing, "answer_start": start_governing}],
                    ),
                    _qa(
                        "MASTER_SERVICES_AGREEMENT__Termination For Convenience",
                        "Termination For Convenience",
                        [{"text": answer_term, "answer_start": start_term}],
                    ),
                    # Answered clause that is NOT in our target set -> should drop.
                    _qa(
                        "MASTER_SERVICES_AGREEMENT__Audit Rights",
                        "Audit Rights",
                        [{"text": "Audit Rights apply.", "answer_start": 0}],
                    ),
                    # Target clause but unanswered -> label_present=False, no span.
                    _qa(
                        "MASTER_SERVICES_AGREEMENT__Non-Compete",
                        "Non-Compete",
                        [],
                    ),
                ],
            }
        ],
    }

    contract_b = {
        "title": "LICENSE_AGREEMENT",
        "paragraphs": [
            {
                "context": "All IP shall be assigned to Licensee. No exclusivity granted.",
                "qas": [
                    # CUAD's "Ip Ownership Assignment" spelling -> mapped to canonical.
                    _qa(
                        "LICENSE_AGREEMENT__Ip Ownership Assignment",
                        "Ip Ownership Assignment",
                        [
                            {
                                "text": "All IP shall be assigned to Licensee.",
                                "answer_start": 0,
                            }
                        ],
                    ),
                    # Two answers for one clause -> two span records.
                    _qa(
                        "LICENSE_AGREEMENT__Exclusivity",
                        "Exclusivity",
                        [
                            {"text": "No exclusivity granted.", "answer_start": 38},
                            {"text": "exclusivity", "answer_start": 41},
                        ],
                    ),
                    # Empty-string answer should be ignored.
                    _qa(
                        "LICENSE_AGREEMENT__Indemnification",
                        "Indemnification",
                        [{"text": "   ", "answer_start": 0}],
                    ),
                ],
            }
        ],
    }

    return {"version": "1", "data": [contract_a, contract_b]}


@pytest.fixture
def cuad_doc() -> dict:
    return _sample_cuad_doc()


# --------------------------------------------------------------------------- #
# extract_clause_type
# --------------------------------------------------------------------------- #


class TestExtractClauseType:
    def test_uses_id_suffix(self):
        qa = {"id": "Foo__Governing Law", "question": "irrelevant"}
        assert extract_clause_type(qa) == "Governing Law"

    def test_falls_back_to_question_regex(self):
        qa = {
            "id": "no-double-underscore-here",
            "question": 'Highlight ... related to "Most Favored Nation" that ...',
        }
        assert extract_clause_type(qa) == "Most Favored Nation"

    def test_normalizes_cuad_capitalization(self):
        qa = {"id": "X__Ip Ownership Assignment"}
        assert extract_clause_type(qa) == "IP Ownership Assignment"

    def test_returns_none_for_off_target_clause(self):
        qa = {"id": "X__Audit Rights"}
        assert extract_clause_type(qa) is None

    def test_returns_none_when_unparseable(self):
        assert extract_clause_type({"id": "no_separator", "question": "blah"}) is None
        assert extract_clause_type({}) is None


# --------------------------------------------------------------------------- #
# parse_cuad
# --------------------------------------------------------------------------- #


class TestParseCuad:
    def test_emits_one_contract_per_entry(self, cuad_doc):
        out = parse_cuad(cuad_doc)
        ids = [c.contract_id for c in out.contracts]
        assert ids == ["MASTER_SERVICES_AGREEMENT", "LICENSE_AGREEMENT"]
        assert all(isinstance(c, ContractRecord) for c in out.contracts)
        assert out.contracts[0].contract_text == CONTRACT_TEXT_A
        assert out.contracts[0].source_file == "MASTER_SERVICES_AGREEMENT"

    def test_emits_label_per_target_clause_per_contract(self, cuad_doc):
        out = parse_cuad(cuad_doc)
        # 8 target clauses * 2 contracts = 16 label rows.
        assert len(out.labels) == len(TARGET_CLAUSE_TYPES) * 2
        assert all(isinstance(l, LabelRecord) for l in out.labels)
        # Every target clause appears for every contract exactly once.
        for contract_id in ("MASTER_SERVICES_AGREEMENT", "LICENSE_AGREEMENT"):
            for clause in TARGET_CLAUSE_TYPES:
                matches = [
                    l for l in out.labels
                    if l.contract_id == contract_id and l.clause_type == clause
                ]
                assert len(matches) == 1, (contract_id, clause)

    def test_label_present_reflects_answers(self, cuad_doc):
        out = parse_cuad(cuad_doc)
        index = {(l.contract_id, l.clause_type): l.label_present for l in out.labels}
        assert index[("MASTER_SERVICES_AGREEMENT", "Governing Law")] is True
        assert index[("MASTER_SERVICES_AGREEMENT", "Termination for Convenience")] is True
        assert index[("MASTER_SERVICES_AGREEMENT", "Non-Compete")] is False
        assert index[("MASTER_SERVICES_AGREEMENT", "Indemnification")] is False
        assert index[("LICENSE_AGREEMENT", "IP Ownership Assignment")] is True
        assert index[("LICENSE_AGREEMENT", "Exclusivity")] is True
        # Whitespace-only answer text should not count as present.
        assert index[("LICENSE_AGREEMENT", "Indemnification")] is False

    def test_spans_only_for_present_clauses(self, cuad_doc):
        out = parse_cuad(cuad_doc)
        assert all(isinstance(s, ClauseSpanRecord) for s in out.spans)
        assert all(s.label_present is True for s in out.spans)
        clause_counts: dict[tuple[str, str], int] = {}
        for s in out.spans:
            clause_counts[(s.contract_id, s.clause_type)] = (
                clause_counts.get((s.contract_id, s.clause_type), 0) + 1
            )
        assert clause_counts[("MASTER_SERVICES_AGREEMENT", "Governing Law")] == 1
        assert clause_counts[("MASTER_SERVICES_AGREEMENT", "Termination for Convenience")] == 1
        assert clause_counts[("LICENSE_AGREEMENT", "IP Ownership Assignment")] == 1
        # Two answers -> two span rows for Exclusivity.
        assert clause_counts[("LICENSE_AGREEMENT", "Exclusivity")] == 2

    def test_span_offsets_match_contract_text(self, cuad_doc):
        out = parse_cuad(cuad_doc)
        text_by_id = {c.contract_id: c.contract_text for c in out.contracts}
        for span in out.spans:
            text = text_by_id[span.contract_id]
            assert span.start_char is not None
            assert span.end_char == span.start_char + len(span.evidence_text)
            assert text[span.start_char:span.end_char] == span.evidence_text

    def test_off_target_clauses_dropped(self, cuad_doc):
        out = parse_cuad(cuad_doc)
        clause_types = {l.clause_type for l in out.labels}
        assert clause_types == set(TARGET_CLAUSE_TYPES)
        for span in out.spans:
            assert span.clause_type in TARGET_CLAUSE_TYPES

    def test_missing_offset_yields_none(self):
        doc = {
            "data": [
                {
                    "title": "X",
                    "paragraphs": [
                        {
                            "context": "Governing law clause.",
                            "qas": [
                                {
                                    "id": "X__Governing Law",
                                    "question": 'related to "Governing Law" foo',
                                    "answers": [{"text": "Governing law clause."}],
                                    "is_impossible": False,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        out = parse_cuad(doc)
        spans = [s for s in out.spans if s.clause_type == "Governing Law"]
        assert len(spans) == 1
        assert spans[0].start_char is None
        assert spans[0].end_char is None

    def test_duplicate_titles_disambiguated(self):
        doc = {
            "data": [
                {
                    "title": "DupTitle",
                    "paragraphs": [{"context": "a", "qas": []}],
                },
                {
                    "title": "DupTitle",
                    "paragraphs": [{"context": "b", "qas": []}],
                },
            ]
        }
        out = parse_cuad(doc)
        ids = [c.contract_id for c in out.contracts]
        assert ids == ["DupTitle", "DupTitle#2"]
        # source_file preserves the original title for both.
        assert all(c.source_file == "DupTitle" for c in out.contracts)

    def test_invalid_root_raises(self):
        with pytest.raises(ValueError, match="data"):
            parse_cuad({"version": "1"})

    def test_empty_target_clauses_raises(self, cuad_doc):
        with pytest.raises(ValueError):
            parse_cuad(cuad_doc, target_clauses=[])

    def test_custom_target_clause_subset(self, cuad_doc):
        out = parse_cuad(cuad_doc, target_clauses=["Governing Law"])
        assert {l.clause_type for l in out.labels} == {"Governing Law"}
        assert all(s.clause_type == "Governing Law" for s in out.spans)
        # Still one contract per entry.
        assert len(out.contracts) == 2

    def test_offset_mismatch_warns_but_does_not_fail(self, caplog):
        # answer_start points at a position whose substring does NOT match.
        doc = {
            "data": [
                {
                    "title": "BAD_OFFSETS",
                    "paragraphs": [
                        {
                            "context": "The agreement is governed by Delaware law.",
                            "qas": [
                                {
                                    "id": "BAD_OFFSETS__Governing Law",
                                    "question": 'related to "Governing Law"',
                                    "answers": [
                                        {"text": "governed by Delaware law", "answer_start": 0}
                                    ],
                                    "is_impossible": False,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        with caplog.at_level("WARNING", logger="contract_question_agent.cuad_loader"):
            out = parse_cuad(doc)
        assert any("Span offset mismatch" in rec.message for rec in caplog.records)
        # Loader still emits the span; validation is advisory only.
        spans = [s for s in out.spans if s.contract_id == "BAD_OFFSETS"]
        assert len(spans) == 1
        assert spans[0].evidence_text == "governed by Delaware law"


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


class TestWriteJsonl:
    def test_writes_pydantic_records(self, tmp_path):
        records = [
            ContractRecord(contract_id="c1", source_file="f1", contract_text="text-1"),
            ContractRecord(contract_id="c2", source_file="f2", contract_text="text-2"),
        ]
        path = tmp_path / "contracts.jsonl"
        n = write_jsonl(path, records)
        assert n == 2
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first == {"contract_id": "c1", "source_file": "f1", "contract_text": "text-1"}

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "out.jsonl"
        write_jsonl(path, [])
        assert path.exists()
        assert path.read_text(encoding="utf-8") == ""

    def test_accepts_plain_dicts(self, tmp_path):
        path = tmp_path / "raw.jsonl"
        write_jsonl(path, [{"a": 1}, {"a": 2}])
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        assert rows == [{"a": 1}, {"a": 2}]


class TestLoadCuadJson:
    def test_loads_plain_json(self, tmp_path, cuad_doc):
        path = tmp_path / "CUAD_v1.json"
        path.write_text(json.dumps(cuad_doc), encoding="utf-8")
        loaded = load_cuad_json(path)
        assert loaded == cuad_doc

    def test_loads_zip_member(self, tmp_path, cuad_doc):
        path = tmp_path / "CUAD_v1.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("CUAD_v1.json", json.dumps(cuad_doc))
        loaded = load_cuad_json(path)
        assert loaded == cuad_doc

    def test_zip_without_json_raises(self, tmp_path):
        path = tmp_path / "empty.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("readme.txt", "no json here")
        with pytest.raises(ValueError, match="No JSON file"):
            load_cuad_json(path)


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #


class TestProcessCuad:
    def test_writes_three_jsonl_files(self, tmp_path, cuad_doc):
        input_path = tmp_path / "CUAD_v1.json"
        input_path.write_text(json.dumps(cuad_doc), encoding="utf-8")
        out_dir = tmp_path / "processed"

        stats = process_cuad(input_path, out_dir)

        contracts_path = out_dir / "contracts.jsonl"
        labels_path = out_dir / "labels.jsonl"
        spans_path = out_dir / "clause_spans.jsonl"
        assert contracts_path.exists()
        assert labels_path.exists()
        assert spans_path.exists()

        contracts = [json.loads(l) for l in contracts_path.read_text("utf-8").splitlines()]
        labels = [json.loads(l) for l in labels_path.read_text("utf-8").splitlines()]
        spans = [json.loads(l) for l in spans_path.read_text("utf-8").splitlines()]

        assert stats.contracts == len(contracts) == 2
        assert stats.labels == len(labels) == len(TARGET_CLAUSE_TYPES) * 2
        assert stats.spans == len(spans)
        assert spans, "expected at least one span row"

        # Spec: every record carries contract_id, source_file, label_present
        # where applicable. Verify those keys appear in their respective files.
        for row in contracts:
            assert {"contract_id", "source_file", "contract_text"} <= row.keys()
        for row in labels:
            assert {
                "contract_id",
                "source_file",
                "clause_type",
                "label_present",
            } <= row.keys()
            assert isinstance(row["label_present"], bool)
        for row in spans:
            assert {
                "contract_id",
                "source_file",
                "clause_type",
                "evidence_text",
                "start_char",
                "end_char",
                "label_present",
            } <= row.keys()
            assert row["label_present"] is True


# --------------------------------------------------------------------------- #
# Sanity checks on module-level constants
# --------------------------------------------------------------------------- #


def test_target_clause_types_match_spec():
    assert TARGET_CLAUSE_TYPES == (
        "Termination for Convenience",
        "Change of Control",
        "Non-Compete",
        "Exclusivity",
        "Most Favored Nation",
        "IP Ownership Assignment",
        "Indemnification",
        "Governing Law",
    )


def test_lookup_covers_every_target():
    for name in TARGET_CLAUSE_TYPES:
        assert any(canonical == name for canonical in CLAUSE_TYPE_LOOKUP.values())
