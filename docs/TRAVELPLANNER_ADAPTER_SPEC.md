# TravelPlanner Adapter v2 Specification

## Scope and frozen boundary

Adapter v2 is a post-hoc, query-independent representation converter. Its only input is the
already-saved raw final answer. It cannot read a TravelPlanner query, task ID, tool trace,
database, reference answer, evaluator verdict, model checkpoint, or validation metric. It does
not call a model. Missing information is represented by `-`; the adapter may not solve, repair,
replace, or supplement a plan.

The frozen transform is therefore:

`raw final answer -> deterministic semantic extraction -> official plan JSON`

## Official contract studied

This specification follows TravelPlanner commit `e52c87f4ac348a3410c46dc3553c519db5ec5e23`:

- `postprocess/parsing.py` requests format conversion with temperature zero.
- `postprocess/openai_request.py` defines the extraction prompt and examples.
- `postprocess/element_extraction.py` extracts the JSON list returned by the parser.
- `postprocess/combination.py` emits one JSONL object containing `idx`, `query`, and `plan`.
- `postprocess/example_evaluation.jsonl` demonstrates the delivered representation.
- `postprocess/format_check.py` invokes official commonsense and hard-constraint code.
- `evaluation/eval.py` computes the official aggregate metrics.

Every day row consumed by the evaluator requires these fields, in this normalized order:

1. `day`
2. `current_city`
3. `transportation`
4. `breakfast`
5. `attraction`
6. `lunch`
7. `dinner`
8. `accommodation`

On a movement day, official examples express `current_city` as `from A to B`. Entity-bearing
meal, attraction, and accommodation values normally use `Name, City`; multiple attractions are
semicolon-separated. `-` is the official absent-value sentinel. Field presence alone is not the
official completeness gate: `is_not_absent` also checks day count and required semantic content.

## Adapter v1 failure

The previous `_labelled_day_blocks` parser recognized day headings but only extracted explicit
`Field:` labels. When a day was written as itinerary prose and no label matched, it executed the
fallback `row["attraction"] = body`. This compressed the entire day narrative—including movement,
meals, lodging, and commentary—into `attraction`.

Consequently v1 lost:

- `current_city`, including the official `from A to B` route representation;
- `transportation`;
- `breakfast`, `lunch`, and `dinner`;
- `accommodation`;
- semantic separation among multiple attractions and other activities.

## Field-by-field difference

| Field | Adapter v1 prose fallback | Official submission / adapter v2 |
|---|---|---|
| `day` | Extracted from heading | Extracted only from an explicit heading or source JSON/table |
| `current_city` | Usually `-` | Explicit label, or explicit `from A to B` movement phrase |
| `transportation` | Usually `-` | Explicit label or explicit transport phrase, preserved |
| `breakfast` | Usually `-` | Explicit `Breakfast:` or `breakfast at ...` only |
| `attraction` | Entire day body | Explicit label or explicit visit/explore/tour/see clause only |
| `lunch` | Usually `-` | Explicit `Lunch:` or `lunch at ...` only |
| `dinner` | Usually `-` | Explicit `Dinner:` or `dinner at ...` only |
| `accommodation` | Usually `-` | Explicit label, `stay at ...`, or `check into ...` only |

## Frozen generic rules

Adapter v2 uses three input-form parsers in fixed order: JSON, Markdown table, then explicit day
blocks. JSON/table keys are mapped only through generic field aliases. Day prose recognizes only
generic travel verbs and field labels. It preserves extracted text apart from whitespace and
surrounding punctuation cleanup. It neither queries the official database nor substitutes entity
names. A city is never appended to a restaurant, attraction, or hotel unless that city is already
inside the extracted source span. It never expands a date range into inferred days.

This logic contains no validation task IDs, validation entities, reference plans, constraint
rules, budget logic, or evaluator feedback. Changes after the one formal evaluation are forbidden
except for a separately documented genuine code-execution bug; metric-guided tuning is forbidden.

## Validity definitions

- **Parser-valid:** the raw answer yields a non-empty plan list.
- **Schema-valid:** every plan row contains exactly the eight required non-empty string fields.
- **Completeness-pass:** the unchanged official evaluator reports `is_not_absent=True`.
- **Official format-check execution:** the unmodified official `postprocess/format_check.py`
  finishes without an exception on the generated submission.

Schema validity is deliberately reported separately from semantic completeness and sandbox
validity; a structurally valid object can still fail official constraints.
