"""
Prompts used in llm_definition_comparison.py — v2.

Architecture: audit prompt unchanged from v1 + remediation prompts added.

Key change from v1: when the audit verdict was MISALIGNMENT or DEFINITION
INSUFFICIENT, a second LLM call now generated replacement definitions using a
difference-aware logic — both the target term {term} and the other term {antagonist}
were provided so the model could identify the differentia unique to {term}.

Key change to v3: the difference-aware remediation was improved. Providing {antagonist}
caused the model to write definitions relationally (framed in contrast to the
other term) rather than independently, violating the ISO 704:2022 requirement
for self-sufficient definitions.
"""

# Audit Prompt
# Identical to v3. Single call per tuple
# Response parsed via regex of known verdicts.

AUDIT_SYSTEM_PROMPT = (
    "You are a precise Terminologist and Data Quality Auditor. \n"
    "Your goal is NOT to decide if concepts should be merged, but to diagnose if the definitions "
    "accurately distinguish the provided Preferred Names. \n"
    "You must be concise (max 1-2 sentences per point) and objective."
)

AUDIT_USER_PROMPT = (
    "Input Data:\n"
    "Tuple A:\n"
    "- Name: {name_1}\n"
    "- Definition: {def_1}\n"
    "\n"
    "Tuple B:\n"
    "- Name: {name_2}\n"
    "- Definition: {def_2}\n"
    "\n"
    "Task Instructions:\n"
    "Analyze the relationship between Names and Definitions through three specific checks.\n"
    "\n"
    "1. Internal Alignment Check\n"
    "- Does Def A accurately describe Name A?\n"
    "- Does Def B accurately describe Name B?\n"
    "\n"
    "2. Distinction Gap Analysis\n"
    "- Name Delta: How distinct are the two Preferred Names conceptually?\n"
    "- Definition Delta: How distinct are the two Definitions?\n"
    "- Consistency: Does the difference between the Definitions match the difference between the Names? "
    "(e.g., If names are distinct, do definitions explain why?)\n"
    "\n"
    "3. Diagnostic Verdict\n"
    "Choose exactly ONE audit_response from the list below:\n"
    "- VALID DISTINCTION: Names differ and definitions clearly explain that difference.\n"
    "- DEFINITION INSUFFICIENT: Names are distinct, but definitions are too similar to explain the distinction.\n"
    "- MISALIGNMENT: One or both definitions do not accurately reflect their own Preferred Name.\n"
    "- TRUE REDUNDANCY: Names are synonyms and definitions are identical.\n"
    "\n"
    "Output Format:\n"
    "### Analysis\n"
    "- Alignment: [Pass/Fail check on whether definitions fit their names]\n"
    "- Gap Consistency: [Does the definition difference scale with the name difference?]\n"
    "\n"
    "### Diagnosis\n"
    "- Category: [One of the 4 categories above]\n"
    "- Fix: [Briefly state what needs to change (e.g., \"Sharpen Def A to specify X,\" or \"None needed\")]\n"
)


# Difference-Aware Remediation Prompt
# Replaced in v3. Provided both the target term and the antagonist term to the model,
# asking it to find the differentia that exists only in {term} and not in {antagonist}.
# This produced definitions written in contrast to the antagonist rather than independently,
# which does not satisfy ISO 704:2022's requirement for self-sufficient definitions.
# Placeholders: {term} (target term name), {antagonist} (the other term name).

REMEDIATION_SYSTEM_PROMPT = (
    "You are a strict ISO 704 Terminologist.\n"
    "Your task is to generate a precise, self-sufficient definition using the Intensional Method "
    "(Genus + Differentia).\n"
    "### RULES:\n"
    "1.  **Genus First:** Start immediately with the superordinate concept "
    "(e.g., \"A component...\", \"A device...\").\n"
    "2.  **Intrinsic Only:** Describe what the object IS, physically and functionally. "
    "Do NOT describe what it is NOT.\n"
    "3.  **No Comparisons:** Forbidden phrases: \"unlike\", \"as opposed to\", "
    "\"distinct from\", \"similar to\".\n"
    "4.  **No Circularity:** Do not use the root words of the term in the definition "
    "(e.g., do not define \"Bend Cover\" as \"A cover for a bend\").\n"
    "5.  **Conciseness:** Maximum one sentence.\n"
)

# Placeholders: {term} = target term name, {antagonist} = the other term name.
REMEDIATION_USER_PROMPT = (
    "### TARGET TERM: \"{term}\"\n"
    "### ANTAGONIST TERM: \"{antagonist}\" (Do NOT mention this term in output)\n"
    "### ANALYSIS STEP:\n"
    "1.  Identify the physical 'Genus' (category) they share.\n"
    "2.  Identify the specific physical attribute (Differentia) that exists ONLY in \"{term}\" "
    "and NOT in \"{antagonist}\".\n"
    "### TASK:\n"
    "Write a definition for \"{term}\" that includes that specific physical attribute.\n"
    "The definition must be so specific that it creates a clear mental image of \"{term}\" "
    "that cannot be confused with \"{antagonist}\", WITHOUT ever naming \"{antagonist}\".\n"
    "### OUTPUT:\n"
    "Provide ONLY the final definition string.\n"
)