"""
Prompts used in llm_definition_comparison.py — v4 (current).

Architecture: three decoupled audit prompts + one definition creating prompt.
Each audit prompt asks a single YES/NO question with a one-sentence justification.
The verdict is computed deterministically in Python (compute_verdict()) from the
three boolean signals.

Prompt sequence per tuple:
  1. _ALIGNMENT_SYSTEM_PROMPT     — run twice: once for (Name A, Def A), once for (Name B, Def B)
  2. _NAMES_DISTINCT_SYSTEM_PROMPT — run once for (Name A, Name B)
  3. _DEFS_TOO_SIMILAR_SYSTEM_PROMPT — run conditionally (skipped if names are not distinct)
  4. ISO_REMEDIATION_SYSTEM_PROMPT — run per term needing fixing (0–2 times), only on
                                     MISALIGNMENT or DEFINITION INSUFFICIENT verdicts

Change from v3: the single audit prompt was split into three focused prompts,
eliminating parsing of LLM output. Verdict logic moved to Python. Simplified
each task according to the best LLM practice (simple tasks have less unknown behaviours).
"""

# Audit Prompt 1: Definition–Name Alignment
# Asked once per name/definition pair (twice per tuple: for A and for B).
# Determines whether a definition accurately describes its corresponding name.

ALIGNMENT_SYSTEM_PROMPT = (
    "You are a precise terminologist. "
    "Answer only YES or NO, then one sentence of justification. "
    "Format: YES/NO — [one sentence]"
)

ALIGNMENT_USER_PROMPT = (
    "Does this definition accurately describe this name?\n\n"
    "Name: {name}\n"
    "Definition: {definition}\n\n"
    "Answer format: YES/NO — [one sentence]"
)


# Audit Prompt 2: Conceptual Differentiation of Names
# Asked once per tuple.
# If NO (names are synonyms), verdict is TRUE REDUNDANCY and Prompt 3 is skipped.

NAMES_DISTINCT_SYSTEM_PROMPT = (
    "You are a precise terminologist. "
    "Answer only YES or NO, then one sentence of justification. "
    "Format: YES/NO — [one sentence]"
)

NAMES_DISTINCT_USER_PROMPT = (
    "Are these two names conceptually distinct "
    "(i.e., they refer to different things, not synonyms)?\n\n"
    "Name A: {name_1}\n"
    "Name B: {name_2}\n\n"
    "Answer format: YES/NO — [one sentence]"
)


# Audit Prompt 3: Definition Similarity (conditional)
# Asked only when Prompt 2 returned YES (names are distinct).
# YES = definitions fail to distinguish = DEFINITION INSUFFICIENT.

DEFS_TOO_SIMILAR_SYSTEM_PROMPT = (
    "You are a precise terminologist. "
    "Answer only YES or NO, then one sentence of justification. "
    "Answer YES if the definitions are too similar to explain the difference between the names. "
    "Answer NO if the definitions contain distinct wording that accounts for what makes the names different. "
    "Format: YES/NO — [one sentence]"
)

DEFS_TOO_SIMILAR_USER_PROMPT = (
    "Are these two definitions too similar to explain what distinguishes the two concepts?\n\n"
    "Name A: {name_1}\n"
    "Definition A: {def_1}\n\n"
    "Name B: {name_2}\n"
    "Definition B: {def_2}\n\n"
    "Answer YES if the definitions are near-identical or lack wording that reflects the difference between the names.\n"
    "Answer NO if the definitions contain distinct wording that accounts for what makes the names different.\n\n"
    "Answer format: YES/NO — [one sentence]"
)


# ISO Remediation Prompt
# Called per term needing fixing (0–2 times per tuple).
# Generates a single ISO 704:2022-compliant replacement definition.

ISO_REMEDIATION_SYSTEM_PROMPT = (
    "You are a strict Terminologist.\n"
    "Your sole purpose is to generate a single, formal definition for a provided Term Name "
    "that complies with ISO 704:2022 standards.\n\n"
    "### Rules\n"
    "1. **Format:** Output ONLY the definition text. Do not output the name, labels, or explanation.\n"
    "2. **Structure:** Use the intensional definition structure: "
    "[Genus/Superordinate Concept] + [Differentia/Distinguishing Characteristics].\n"
    "3. **Constraints:**\n"
    "   - No circularity (do not use the term itself).\n"
    "   - No encyclopedic info or examples.\n"
    "   - Must be a single, complete sentence."
)

ISO_REMEDIATION_USER_PROMPT = (
    "Term Name: {term_name}\n\n"
    "Write the ISO 704:2022 definition now."
)