# Contract Verification Question Skill

## Purpose

This skill converts contract clause excerpts into structured verification questions for human and professional review.

It should reduce information asymmetry by identifying:

- unknowns
- decision risks
- selected review lenses
- legal review questions
- verification questions

## Responsibility Boundaries

The skill should:

- generate questions, not legal answers
- support human/professional review
- stay grounded in the given clause text
- use structured output

The skill must not:

- Do not provide legal advice
- decide whether a clause is legal, illegal, enforceable, unenforceable, valid, invalid, acceptable, or unacceptable
- recommend signing or not signing
- treat tool output as legal conclusions

## MCP Candidate Lens Usage

MCP provides candidate review lenses. The application may retrieve these lenses deterministically before generation.

The model should select only lenses relevant to the clause text. Unsupported or irrelevant lenses should be ignored.

Selected MCP-derived lenses should use:

```text
source = "mcp_clause_review_hints"
```

Candidate lenses are advisory context, not conclusions.

## Selected Review Lenses

`selected_review_lenses` should make the model's use of candidate lenses observable.

Each selected lens should include:

- label
- source
- reason

Reasons should be short, grounded in the clause text, and should not include legal conclusions.

## Output Expectations

Expected output categories are:

- selected_review_lenses
- unknowns
- decision_risks
- legal_review_questions
- verification_questions
- suggested_next_step
- safety_disclaimer
- safety_status / safety_warnings

Plain string fields should avoid Markdown list markers or blockquote markers.

Generated plain strings may be normalized with NFKC and marker cleanup.

## Observability Expectations

Runs should make visible:

- whether MCP hints were enabled
- whether lookup was attempted
- whether hints were found
- how many candidate hints were provided
- how many selected review lenses were produced

Do not log raw contract evidence text in telemetry summaries.

## Graceful Degradation

If MCP hints are unavailable, missing, or not found:

- generation should still proceed
- selected_review_lenses may be empty
- the agent should still generate useful verification questions from the clause text
- fallback/retry behavior will be handled separately
