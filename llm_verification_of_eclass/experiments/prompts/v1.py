"""
Prompts used in llm_definition_comparison.py — v1.

Architecture: structured audit prompt only, no remediation step.

The first version of the experiment. One LLM call per tuple assessed
alignment, distinctiveness, and produced a verdict.

Key change to v2: remediation was added. When the audit identified MISALIGNMENT
or DEFINITION INSUFFICIENT, a second LLM call suggested newly generated definitions.
The audit prompt itself was unchanged between v1 and v3.
"""

# Audit Prompt
# One call per tuple. Asks the model to perform all three reasoning steps
# (alignment, gap analysis, verdict) in one structured completion.
# Output is parsed via regex to extract the Category and Fix fields.

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