# Prompt Evolution — `llm_definition_comparison`

This file documents the evolution of prompts used in Experiment 5.
Each version corresponds to a change in prompting strategy, with the corresponding
code changes in `llm_definition_comparison.py`.

---

## v1 — Single complex audit prompt

**Prompt file:** `v1.py`

**Prompts:** `AUDIT_SYSTEM_PROMPT`, `AUDIT_USER_PROMPT`

**What it did:** A single LLM call per tuple assessed the name–definition pairs
through three structured steps (alignment, gap analysis, verdict) and produced one
of four diagnostic categories. 

**Rationale for change → v2:** The changes identified could be used for better definitions. The audit prompt itself was kept unchanged.

---

## v2 — Audit prompt unchanged + antagonist-aware remediation added

**Prompt file:** `v2.py`

**Prompts:** `AUDIT_SYSTEM_PROMPT`, `AUDIT_USER_PROMPT` (identical to v1),
`REMEDIATION_SYSTEM_PROMPT`, `REMEDIATION_USER_PROMPT`

**What it did:** When the audit verdict was MISALIGNMENT or DEFINITION INSUFFICIENT,
a second LLM call generated replacement definitions using a difference-aware logic:
both the target term `{term}` and the other term `{antagonist}` were provided, and the
model was asked to identify the differentia that exists only in `{term}`.

**Problem:** The difference-aware approach caused the model to write definitions
as comparisons, framing each concept in contrast to the other. This violates the ISO 704:2022 requirement for self-sufficient
definitions and produced outputs that read as comparisons rather than standalone
definitions.

**Rationale for change → v3:** Replace the difference-aware remediation with an 
approach that generates each definition in isolation. Audit unchanged.

---

## v3 — Structured audit + intrinsic remediation

**Prompt file:** `v3.py`

**Prompts:** `AUDIT_SYSTEM_PROMPT`, `AUDIT_USER_PROMPT`, `ISO_REMEDIATION_SYSTEM_PROMPT`,
`ISO_REMEDIATION_USER_PROMPT`

**What it did:** The audit prompt was unchanged from v2. The remediation prompt was
replaced entirely: the antagonist term was removed, and the model was given only the target
`{term_name}` and asked to produce a definition following the ISO 704:2022 intensional
method (Genus + Differentia) as a self-sufficient sentence.

**Problem:** The audit was still a single LLM call producing structured but complex response that
required regex parsing to extract the verdict and the fix field. This parsing was
fragile on edge cases (especially in identifying which term needs to change), and a single prompt was responsible
for three distinct reasoning tasks (alignment, distinctiveness, definitional sufficiency) in one response, making it
harder for the model to handle each cleanly.

**Rationale for change → v4:** Decouple the audit into three focused YES/NO prompts
so that each reasoning step is isolated. The responses are not parsed thanks to only two possible scenarios: YES or NO.

---

## v4 — Decoupled audit into three focused prompts (current)

**Prompt file:** `v4.py`

**Prompts:** `ALIGNMENT_SYSTEM_PROMPT`, `ALIGNMENT_USER_PROMPT`,
`NAMES_DISTINCT_SYSTEM_PROMPT`, `NAMES_DISTINCT_USER_PROMPT`,
`DEFS_TOO_SIMILAR_SYSTEM_PROMPT`, `DEFS_TOO_SIMILAR_USER_PROMPT`,
`ISO_REMEDIATION_SYSTEM_PROMPT`, `ISO_REMEDIATION_USER_PROMPT`

**What it did:** The single audit prompt was split into three prompts, simplifying the task and making responses of LLM more predictable,
each asking a single YES/NO question with a mandatory one-sentence reasoning. The verdict is
computed deterministically in Python (`compute_verdict()`) from the three boolean
signals returned by these prompts. Prompt 3 (definition similarity) is skipped
when Prompt 2 determines the names are not distinct (TRUE REDUNDANCY path). 
The ISO remediation prompt is unchanged from v3.

| Prompt | Question | Calls per tuple |
|---|---|---|
| `ALIGNMENT_*` | Does this definition accurately describe this name? | 2 (once for A, once for B) |
| `NAMES_DISTINCT_*` | Are these two names conceptually distinct? | 1 |
| `DEFS_TOO_SIMILAR_*` | Are the definitions too similar to explain the name distinction? | 0–1 (skipped if names not distinct) |
| `ISO_REMEDIATION_*` | Generate an ISO 704:2022 definition for this term. | 0–2 (only on MISALIGNMENT or DEFINITION INSUFFICIENT) |

