"""
Experiment 5: Analyze name-definition tuple pairs using Llama 3.1 via Ollama.

This module helps to identify inconsistencies in definitions by comparing tuple pairs.
"""

import argparse
import csv
import logging
import os
import re
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import icontract

from openai import OpenAI

# Import LoggerFactory from the common module
import sys
sys.path.append(str(Path(__file__).parent.parent))
from llm_verification_of_eclass.common.logger import LoggerFactory


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM backend.

    :ivar model: Name of the Ollama model to use
    :ivar temperature: Sampling temperature (0.0 for deterministic)
    :ivar base_url: Ollama API endpoint URL
    :ivar context_window: Optional context window size
    """

    model: str
    temperature: float = 0.0
    base_url: Optional[str] = None
    context_window: Optional[int] = None


@icontract.invariant(lambda self: self.config is not None, "Configuration must always be present")
class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @icontract.require(lambda config: config is not None, "Config cannot be None")
    def __init__(self, config: OllamaConfig) -> None:
        """Initialize the LLM client.

        :param config: Configuration for the LLM backend
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration.

        :raise ConnectionError: If backend is not accessible
        :raise ValueError: If configuration is invalid
        """
        raise NotImplementedError()

    @abstractmethod
    @icontract.require(lambda system_prompt: len(system_prompt.strip()) > 0, "System prompt cannot be empty")
    @icontract.require(lambda user_prompt: len(user_prompt.strip()) > 0, "User prompt cannot be empty")
    @icontract.ensure(lambda result: isinstance(result, str) and len(result) > 0, "Must return a non-empty string")
    def create_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Create a chat completion.

        :param system_prompt: System instruction for the LLM
        :param user_prompt: User query/task
        :return: The LLM's response text
        """
        raise NotImplementedError()


class OllamaClient(LLMClient):
    """Ollama implementation."""

    def __init__(self, config: OllamaConfig) -> None:
        """Initialize Ollama client.

        :param config: Ollama configuration
        """
        super().__init__(config)
        self.client = OpenAI(
            base_url=config.base_url or "http://localhost:11434/v1",
            api_key="ollama",  # Dummy key required by OpenAI client
        )

    def _validate_config(self) -> None:
        """Validate that Ollama is running and accessible.

        Precondition: Ollama must be running with the model pulled.

        :raise ConnectionError: If cannot connect to Ollama
        """
        try:
            # Check if Ollama is listening on the expected port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", 11434))
            sock.close()
            if result != 0:
                raise ConnectionError(
                    "Cannot connect to Ollama at localhost:11434. "
                    "Please ensure 'ollama serve' is running."
                )
        except Exception as e:
            raise ConnectionError(f"Ollama validation failed: {e}")

    def create_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama via OpenAI-compatible endpoint.

        :param system_prompt: System instruction
        :param user_prompt: User query
        :return: LLM response text
        """
        kwargs = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        # Add Ollama options if context window specified
        if self.config.context_window:
            kwargs["options"] = {"num_ctx": self.config.context_window}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


STRUCTURAL_DEFINITIONS = {
    "Sub-group (4th level) for objects that cannot be classified into other specified sub-groups in the existing structure, but that are classified to their parent-class on the 3rd level",
    "Group (3rd level) for objects that cannot be classified into other specified groups in the existing structure, but that are classified to their parent-class on the 2nd level. A xx-xx-90-00-class cannot have any other sub-groups besides the generic xx-xx-90-90-class (…(Other, unspecified)) to be generally valid within the parent class on the 2nd level",
    "sub-group (4th level) for objects that cannot be classified into other specified sub-groups in the existing structure, but that are classified to their parent-class on the 3rd level",
    "Sub-group (4th level) for objects with complemental characteristics, without which the basic function of the objects classified under the parent-class on the 3rd level is still guaranteed",
    "Sub-group (4th level) for objects with spare part characteristics that maintain or re-store the unfiltered condition of the objects classified under the parent-class on the 3rd level",
    "sub-group (4th level) for objects with spare part characteristics that maintain or re-store the unfiltered condition of the objects classified under the parent-class on the 3rd level",
    "sub-group (4th level) for objects with complemental characteristics, without which the basic function of the objects classified under the parent-class on the 3rd level is still guaranteed",
    "Subgroup (4th level) for objects that cannot be classified in other specified subgroups of the existing structure but are classified to their 3rd level parent class",
    "Subgroup (fourth level) for objects which cannot be classified in other subgroups specified in the existing structure, but which are classified in the parent class at the third level",
    "Group (3rd level) for objects that cannot be classified in other specified groups in the existing structure, but are classified to their 2nd level parent class. A xx-xx-90-00 class cannot have any other subgroups except the generic xx-xx-90-90 class (...(Other, unspecified)) to be generally valid within the 2nd level parent class.",
    "Sub-group (4th level) for objects with additional characteristics, without which the basic function of the objects classified under the parent class of the 3rd level is still guaranteed",
    "Sub-group (4th level) for objects that cannot be classified into other specified sub-groups in the existing structure, but that are classified to their parent-class on the 3rd Level",
    "Group (3rd level) for objects which can not be put in order into other groups of the existing structure but are assigned to the main group lying over this at the 2nd level and are applied in a quite specific field of application",
    "Sub-group (4th level) for objects with spare part characteristics that maintain or restore the unfiltered condition of the objects classified under the parent-class on the 3rd level",
    "Sub-group (4th level) for objects with spare part characteristics that maintain or re-store the unfiltered condition of the objects classified under the parent-class on the 3rd Level",
    "group (3rd level) for objects that cannot be classified into other specified groups in the existing structure, but that are classified to their parent-class on the 2nd level. a xx-xx-90-00-class cannot have any other sub-groups besides the generic xx-xx-90-90-class (…(other, unspecified)) to be generally valid within the parent class on the 2nd level",
    "group (3rd level) for objects that cannot be classified into other specified groups in the existing structure, but that are classified to their parent-class on the 2nd level. A xx-xx-90-00-class cannot have any other sub-groups besides the generic xx-xx-90-90-class (…(Other, unspecified)) to be generally valid within the parent class on the 2nd level",
    "Sub-group (4th level) for objects with complemental characteristics, without which the basic function of the objects classified under the parent-class on the 3rd level is still guaranteedel",
    "Subgroup (4th level) for objects that cannot be classified into other specified subgroups of the existing structure, but that are classified to their parent-class on the 3rd level",
    "Subgroup (4th level) for objects with spare part characteristics that maintain or restore the unfiltered condition of the objects classified under the parent-class on the 3rd level",
    "Subgroup (4th level) for objects with complemental characteristics without which the basic function of the objects classified under the parent-class on the 3rd level is still guaranteed",
    "Subclass (4th level) for objects with spare part characteristics that maintain or restore the original condition of the objects classified under the parent class of the 3rd level",
    "Sub-group (4th level) for objects with spare part characteristics that maintain or restore the original condition of the objects classified under the parent-class on the 3rd level",
    "Sub-group (4th level) for objects with spare part characteristics that maintain or re-store the original condition of the objects classified under the parent-class on the 3rd Level",
    "Sub-group (4th level) for objects with spare part characteristics that maintain or re-store the original condition of the objects classified under the parent-class on the 3rd level",
    "Group (3rd level) for objects that cannot be classified into other specified groups in the existing structure, but that are classified to their parent-class on the 2nd level. A xx-xx-90-00-class cannot have any other sub-groups besides the generic xx-xx-90-90-class (â€¦(Other, unspecified)) to be generally valid within the parent class on the 2nd level",
    "sub-group (4th level) for objects with spare part characteristics that maintain or re-store the original condition of the objects classified under the parent-class on the 3rd level",
    "Subgroup (4th level) for objects with spare part characteristics that maintain or restore the original condition of the objects classified under the parent-class on the 3rd level"
}

CHEMICAL_COMPOUNDS = [
    "benz", "phenyl", "sulfon", "sulfonate", "sulfate", "sulfide", "sulfite", "sulfonyl", "nitro", "nitrate", "amino",
    "imino", "carbox", "carbonyl", "aminocarbonyl", "carboxyl", "carboxylate", "amide", "amidate", "acetate",
    "propionate", "butyrate", "oxalate", "lactate", "phosphate", "hydroxide", "oxide", "peroxide", "cyano", "oxo",
    "azo", "triazine", "pyridine", "pyrimidine", "imidazole", "quinazoline", "indole", "naphth", "anthrac", "vinyl",
    "vinylen", "ethyl", "propyl", "butyl", "methyl", "hydroxy", "methoxy", "phenoxy", "benzoyl", "benzyl", "ylidene",
    "ylide", "yl", "anilino", "naphthyl", "anthryl", "chloro", "bromo", "fluoro", "dichloro", "trichloro", "chloride",
    "chlorophenyl", "dichlorophenyl", "trifluoromethyl", "sodium", "potassium", "ammonium", "lithium", "magnesium",
    "salts", "hepta", "hexa", "penta", "tetra", "tri", "di", "mono", "bis", "tris", "aluminium", "paraffins",
    "phosphating", "acrylates", "hexafluoride", "arenes", "polymethyl", "maltene", "sulfate", "sulphooxy",
    "terephthalat", "phosphonat", "sulphate", "polyolefin", "onhydrochlorid"
]


def is_structural_definition(text_a: str, text_b: str) -> bool:
    """
    Checks if text_a or text_b matches an entry in the structural definitions set.
    """
    if text_a in STRUCTURAL_DEFINITIONS:
        return True
    if text_b in STRUCTURAL_DEFINITIONS:
        return True
    return False


def is_chemical_definition(text_a: str, text_b: str) -> bool:
    """
    Checks if any chemical keyword is present as a substring in text_a or text_b.
    The check is case-insensitive.
    """
    # Convert inputs to lowercase to ensure 'Sodium' matches 'sodium'
    a_lower = text_a.lower() if text_a else ""
    b_lower = text_b.lower() if text_b else ""

    # Check if any chemical keyword exists inside text_a
    if any(keyword in a_lower for keyword in CHEMICAL_COMPOUNDS):
        return True

    # Check if any chemical keyword exists inside text_b
    if any(keyword in b_lower for keyword in CHEMICAL_COMPOUNDS):
        return True

    return False


@dataclass
class AuditSignals:
    """Structured results from the three focused audit prompts.

    :ivar alignment_a: Whether Definition A accurately describes Name A
    :ivar alignment_b: Whether Definition B accurately describes Name B
    :ivar alignment_a_justification: One-sentence justification for alignment_a
    :ivar alignment_b_justification: One-sentence justification for alignment_b
    :ivar names_distinct: Whether the two names are conceptually distinct
    :ivar names_distinct_justification: One-sentence justification for names_distinct
    :ivar defs_explain_distinction: Whether the definitions explain the distinction
                                    (None when names_distinct is False — prompt skipped)
    :ivar defs_explain_distinction_justification: Justification, or None when skipped
    """

    alignment_a: bool
    alignment_b: bool
    alignment_a_justification: str
    alignment_b_justification: str
    names_distinct: bool
    names_distinct_justification: str
    defs_explain_distinction: Optional[bool]
    defs_explain_distinction_justification: Optional[str]


def _parse_yes_no_response(response: str) -> Tuple[bool, str]:
    """Parse a YES/NO — [justification] response from the LLM.

    :param response: Raw LLM response text
    :return: Tuple of (bool answer, justification string)
    :raise ValueError: If response does not start with YES or NO
    """
    stripped = response.strip()
    upper = stripped.upper()

    if upper.startswith("YES"):
        answer = True
    elif upper.startswith("NO"):
        answer = False
    else:
        raise ValueError(f"Expected YES/NO response, got: {stripped!r}")

    # Extract justification after the delimiter (dash, colon, or whitespace)
    remainder = stripped[3:].lstrip(" —–-:")
    justification = remainder.strip() if remainder.strip() else "(no justification provided)"

    return answer, justification


_ALIGNMENT_SYSTEM_PROMPT = (
    "You are a precise terminologist. "
    "Answer only YES or NO, then one sentence of justification. "
    "Format: YES/NO — [one sentence]"
)

_NAMES_DISTINCT_SYSTEM_PROMPT = (
    "You are a precise terminologist. "
    "Answer only YES or NO, then one sentence of justification. "
    "Format: YES/NO — [one sentence]"
)

_DEFS_EXPLAIN_SYSTEM_PROMPT = (
    "You are a precise terminologist. "
    "Answer only YES or NO, then one sentence of justification. "
    "Format: YES/NO — [one sentence]"
)


def _check_alignment(client: LLMClient, name: str, definition: str) -> Tuple[bool, str]:
    """Prompt 1 — ask whether a single definition accurately describes its name.

    :param client: LLM client
    :param name: Preferred name of the concept
    :param definition: Definition text to evaluate
    :return: Tuple of (alignment result, justification)
    """
    user_prompt = (
        "Does this definition accurately describe this name?\n\n"
        f"Name: {name}\n"
        f"Definition: {definition}\n\n"
        "Answer format: YES/NO — [one sentence]"
    )
    response = client.create_completion(_ALIGNMENT_SYSTEM_PROMPT, user_prompt)
    return _parse_yes_no_response(response)


def _check_names_distinct(client: LLMClient, name_1: str, name_2: str) -> Tuple[bool, str]:
    """Prompt 2 — ask whether two names are conceptually distinct.

    :param client: LLM client
    :param name_1: First preferred name
    :param name_2: Second preferred name
    :return: Tuple of (distinct result, justification)
    """
    user_prompt = (
        "Are these two names conceptually distinct "
        "(i.e., they refer to different things, not synonyms)?\n\n"
        f"Name A: {name_1}\n"
        f"Name B: {name_2}\n\n"
        "Answer format: YES/NO — [one sentence]"
    )
    response = client.create_completion(_NAMES_DISTINCT_SYSTEM_PROMPT, user_prompt)
    return _parse_yes_no_response(response)


def _check_defs_explain_distinction(
    client: LLMClient,
    name_1: str, def_1: str,
    name_2: str, def_2: str,
) -> Tuple[bool, str]:
    """Prompt 3 — ask whether definitions sufficiently explain the distinction between names.

    Only called when names are already confirmed distinct.

    :param client: LLM client
    :param name_1: First preferred name
    :param def_1: First definition
    :param name_2: Second preferred name
    :param def_2: Second definition
    :return: Tuple of (explanation result, justification)
    """
    user_prompt = (
        "Do these two definitions sufficiently explain what distinguishes "
        "the two concepts from each other?\n\n"
        f"Name A: {name_1}\n"
        f"Definition A: {def_1}\n\n"
        f"Name B: {name_2}\n"
        f"Definition B: {def_2}\n\n"
        "Answer format: YES/NO — [one sentence]"
    )
    response = client.create_completion(_DEFS_EXPLAIN_SYSTEM_PROMPT, user_prompt)
    return _parse_yes_no_response(response)


def run_audit_prompts(
    client: LLMClient,
    name_1: str, def_1: str,
    name_2: str, def_2: str,
    logger: logging.Logger,
) -> AuditSignals:
    """Run all three focused audit prompts and return structured signals.

    Prompt 3 is skipped when names are not distinct (TRUE REDUNDANCY path).

    :param client: LLM client
    :param name_1: First preferred name
    :param def_1: First definition
    :param name_2: Second preferred name
    :param def_2: Second definition
    :param logger: Logger instance
    :return: AuditSignals with results from each prompt
    """
    alignment_a, just_a = _check_alignment(client, name_1, def_1)
    logger.info(f"  [Prompt 1a] Alignment A: {'YES' if alignment_a else 'NO'} — {just_a}")

    alignment_b, just_b = _check_alignment(client, name_2, def_2)
    logger.info(f"  [Prompt 1b] Alignment B: {'YES' if alignment_b else 'NO'} — {just_b}")

    names_distinct, just_names = _check_names_distinct(client, name_1, name_2)
    logger.info(f"  [Prompt 2]  Names distinct: {'YES' if names_distinct else 'NO'} — {just_names}")

    if names_distinct:
        defs_explain, just_defs = _check_defs_explain_distinction(
            client, name_1, def_1, name_2, def_2
        )
        logger.info(f"  [Prompt 3]  Defs explain distinction: {'YES' if defs_explain else 'NO'} — {just_defs}")
    else:
        defs_explain = None
        just_defs = None
        logger.info("  [Prompt 3]  Skipped (names not distinct)")

    return AuditSignals(
        alignment_a=alignment_a,
        alignment_b=alignment_b,
        alignment_a_justification=just_a,
        alignment_b_justification=just_b,
        names_distinct=names_distinct,
        names_distinct_justification=just_names,
        defs_explain_distinction=defs_explain,
        defs_explain_distinction_justification=just_defs,
    )


def compute_verdict(signals: AuditSignals) -> str:
    """Derive a deterministic verdict from the three audit signals.

    Decision table:
    - Any alignment fails                      → MISALIGNMENT
    - Both align, names not distinct           → TRUE REDUNDANCY
    - Both align, names distinct, defs weak    → DEFINITION INSUFFICIENT
    - Both align, names distinct, defs strong  → VALID DISTINCTION

    :param signals: Populated AuditSignals instance
    :return: One of the four verdict strings
    """
    if not signals.alignment_a or not signals.alignment_b:
        return "MISALIGNMENT"
    if not signals.names_distinct:
        return "TRUE REDUNDANCY"
    if not signals.defs_explain_distinction:
        return "DEFINITION INSUFFICIENT"
    return "VALID DISTINCTION"


def _terms_to_fix_from_signals(signals: AuditSignals) -> List[str]:
    """Derive which definitions need remediation directly from audit signals.

    Replaces the fragile string-parsing of the old _parse_fix_section.

    :param signals: Populated AuditSignals instance
    :return: List of 'A', 'B', both, or empty
    """
    verdict = compute_verdict(signals)
    if verdict == "MISALIGNMENT":
        terms = []
        if not signals.alignment_a:
            terms.append("A")
        if not signals.alignment_b:
            terms.append("B")
        return terms
    if verdict == "DEFINITION INSUFFICIENT":
        return ["A", "B"]
    return []


def generate_iso_definition(client: LLMClient, term_name: str, system_prompt: str, user_prompt_template: str) -> str:
    """Generate an ISO 704:2022 compliant definition for a term.

    :param client: LLM client to use for generation
    :param term_name: The name of the term to define
    :param system_prompt: System instruction for the LLM
    :param user_prompt_template: User prompt template with {term_name} placeholder
    :return: Generated definition text
    """
    user_prompt = user_prompt_template.format(term_name=term_name)
    definition = client.create_completion(system_prompt, user_prompt)
    return definition.strip()


def remediate_definitions(
    client: LLMClient,
    name_1: str,
    def_1: str,
    name_2: str,
    def_2: str,
    signals: AuditSignals,
    iso_system_prompt: str,
    iso_user_prompt_template: str,
    logger: logging.Logger,
) -> Dict[str, str]:
    """Generate improved definitions based on identified issues.

    :param client: LLM client to use for generation
    :param name_1: First name
    :param def_1: First definition
    :param name_2: Second name
    :param def_2: Second definition
    :param signals: Structured audit signals from the three focused prompts
    :param iso_system_prompt: System prompt for ISO definition generation
    :param iso_user_prompt_template: User prompt template for ISO definition generation
    :param logger: Logger instance
    :return: Dictionary mapping 'A' and/or 'B' to new definitions
    """
    # Derive which definitions need fixing directly from signals — no string parsing
    terms_to_fix = _terms_to_fix_from_signals(signals)

    if not terms_to_fix:
        return {}

    new_definitions: Dict[str, str] = {}

    # Generate new definition for each identified term
    for term in terms_to_fix:
        if term == "A":
            term_name = name_1
            term_label = "Definition A"
        else:  # term == "B"
            term_name = name_2
            term_label = "Definition B"

        logger.info(f"  Generating ISO 704 definition for '{term_name}'...")

        try:
            new_def = generate_iso_definition(client, term_name, iso_system_prompt, iso_user_prompt_template)
            new_definitions[term] = new_def
            logger.info(f"---> {term_label}: {new_def}")
        except Exception as e:
            logger.error(f"---> Failed to generate {term_label}: {e}")

    return new_definitions


def analyze_tuple(
    client: LLMClient,
    tuple_data: List[str],
    tuple_id: int,
    iso_system_prompt: str,
    iso_user_prompt_template: str,
    logger: logging.Logger,
    enable_remediation: bool = True,
) -> Dict:
    """Analyze a single name-definition pair using three focused audit prompts.

    :param client: LLM client to use for analysis
    :param tuple_data: List containing [name_1, def_1, name_2, def_2]
    :param tuple_id: Number of this tuple
    :param iso_system_prompt: System prompt for ISO definition generation
    :param iso_user_prompt_template: User prompt template for ISO definition generation
    :param logger: Logger instance for logging
    :param enable_remediation: Whether to generate improved definitions
    :return: Dictionary with analysis results and proposed fixes
    :raise ValueError: If tuple_data does not contain exactly 4 elements
    """
    if len(tuple_data) != 4:
        raise ValueError(
            f"Tuple {tuple_id} has {len(tuple_data)} elements, expected 4"
        )

    name_1, def_1, name_2, def_2 = tuple_data

    logger.info(f"Tuple number {tuple_id} will be assessed:")
    logger.info(f"\tName 1: {name_1}")
    logger.info(f"\tDef 1: {def_1}")
    logger.info(f"\tName 2: {name_2}")
    logger.info(f"\tDef 2: {def_2}")
    logger.info("")

    try:
        # Run the three focused audit prompts
        signals = run_audit_prompts(client, name_1, def_1, name_2, def_2, logger)

        # Derive verdict from signals — no LLM call, no string parsing
        verdict = compute_verdict(signals)
        logger.info(f"  [Verdict]   {verdict}")
        logger.info("\n" + "=" * 80 + "\n")

        result = {
            "tuple_id": tuple_id,
            "name_1": name_1,
            "name_2": name_2,
            "audit_signals": signals,
            "audit_response": verdict,  # kept for CSV compatibility
            "status": "success",
        }

        # region Remediation
        needs_fix = verdict in ("MISALIGNMENT", "DEFINITION INSUFFICIENT")
        if enable_remediation and needs_fix:
            logger.info("\n--- Remediation Part ---")

            try:
                new_definitions = remediate_definitions(
                    client=client,
                    name_1=name_1,
                    def_1=def_1,
                    name_2=name_2,
                    def_2=def_2,
                    signals=signals,
                    iso_system_prompt=iso_system_prompt,
                    iso_user_prompt_template=iso_user_prompt_template,
                    logger=logger,
                )

                if new_definitions:
                    logger.info("")
                    result["proposed_definitions"] = new_definitions
                else:
                    logger.info("  No specific definitions identified for remediation.")
                    result["proposed_definitions"] = {}

            except Exception as e:
                logger.error(f"  Remediation failed: {e}")
                result["proposed_definitions"] = {}
        else:
            result["proposed_definitions"] = {}
        # endregion

        logger.info("\n" + "=" * 15 + "\n")

        return result

    except Exception as e:
        logger.error(f"Error processing tuple {tuple_id}: {e}")
        return {
            "tuple_id": tuple_id,
            "name_1": name_1,
            "name_2": name_2,
            "audit_signals": None,
            "audit_response": None,
            "status": "error",
            "error": str(e),
            "proposed_definitions": {},
        }


def config_from_env() -> OllamaConfig:
    """Create Ollama configuration from environment variables.

    Environment Variables:
    - LLM_MODEL: Model name (default: "llama3.1")
    - OLLAMA_BASE_URL: Ollama base URL (default: "http://localhost:11434/v1")
    - LLM_CONTEXT_WINDOW: Context window size (optional)

    :return: Ollama configuration
    """
    model = os.getenv("LLM_MODEL", "llama3.1")
    base_url = os.getenv("OLLAMA_BASE_URL")

    context_window_str = os.getenv("LLM_CONTEXT_WINDOW")
    context_window = int(context_window_str) if context_window_str else None

    return OllamaConfig(
        model=model, base_url=base_url, context_window=context_window
    )


def remove_id_from_preferred_name(preferred_name: str) -> str:
    """Remove ID pattern (e.g., (0173-1#01-AGF076#008)) from preferred name.

    Finds the last closing bracket, finds its matching opening bracket,
    and removes that entire pattern from the end of the string.

    :param preferred_name: The preferred name possibly with ID
    :return: Clean preferred name without ID
    """
    # Find last closing bracket
    last_close = preferred_name.rfind(')')
    if last_close == -1:
        return preferred_name.strip()

    # Find matching opening bracket by counting backwards
    bracket_count = 1
    pos = last_close - 1
    while pos >= 0 and bracket_count > 0:
        if preferred_name[pos] == ')':
            bracket_count += 1
        elif preferred_name[pos] == '(':
            bracket_count -= 1
        pos -= 1

    # pos+1 is now at the opening bracket
    if bracket_count == 0:
        return preferred_name[:pos+1].strip()

    return preferred_name.strip()


def extract_verdict(audit_response: str) -> str:
    """Extract the verdict from the audit response.

    With the decoupled audit, audit_response is already the verdict string
    produced by compute_verdict(). This function is kept for CSV compatibility
    and as a safety net for UNKNOWN responses.

    :param audit_response: The verdict string (or empty/None on error)
    :return: One of the 4 verdicts or "UNKNOWN" if not found
    """
    verdicts = ["VALID DISTINCTION", "DEFINITION INSUFFICIENT", "MISALIGNMENT", "TRUE REDUNDANCY"]

    audit_upper = audit_response.upper()
    for verdict in verdicts:
        if verdict in audit_upper:
            return verdict

    return "UNKNOWN"


def load_csv_data(csv_path: Path, logger: logging.Logger) -> List[Tuple[str, str, str, str, str]]:
    """Load data from CSV file.

    :param csv_path: Path to the CSV file
    :param logger: Logger instance
    :return: List of tuples (distance, preferred_names_a, text_a, preferred_names_b, text_b)
    """
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append((
                row['distance'],
                row['preferred_names_a'],
                row['text_a'],
                row['preferred_names_b'],
                row['text_b']
            ))

    logger.info(f"Loaded {len(data)} rows from {csv_path}")
    return data


def save_results_to_csv(results: List[Dict], output_path: Path, logger: logging.Logger) -> None:
    """Save analysis results to CSV file.

    :param results: List of result dictionaries
    :param output_path: Path to output CSV file
    :param logger: Logger instance
    """
    if not results:
        logger.warning("No results to save")
        return

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = [
            'distance', 'preferred_names_a', 'text_a', 'preferred_names_b', 'text_b',
            'verdict', 'llm_definition_a', 'llm_definition_b'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result)

    logger.info(f"Saved {len(results)} results to {output_path}")


def main() -> List[Dict]:
    """Execute the main analysis workflow.

    :return: List of analysis results for all tuples
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze definition tuples using LLM")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "real"],
        required=True,
        help="Mode to run: 'test' for test samples, 'real' for CSV data"
    )
    args = parser.parse_args()

    # Create experiment-5 directory if it doesn't exist
    script_dir = Path(__file__).parent
    experiment_dir = script_dir / "experiment-5"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = experiment_dir / f"{timestamp}_experiment.log"

    logger = LoggerFactory.get_logger(__name__)
    file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Define all prompts
    # Audit prompts are module-level constants (_ALIGNMENT_SYSTEM_PROMPT,
    # _NAMES_DISTINCT_SYSTEM_PROMPT, _DEFS_EXPLAIN_SYSTEM_PROMPT) used directly
    # by run_audit_prompts(). Only the ISO remediation prompts are passed around.

    iso_system_prompt = """You are a strict Terminologist. 
Your sole purpose is to generate a single, formal definition for a provided Term Name that complies with ISO 704:2022 standards.

### Rules
1. **Format:** Output ONLY the definition text. Do not output the name, labels, or explanation.
2. **Structure:** Use the intensional definition structure: [Genus/Superordinate Concept] + [Differentia/Distinguishing Characteristics].
3. **Constraints:**
   - No circularity (do not use the term itself).
   - No encyclopedic info or examples.
   - Must be a single, complete sentence."""

    iso_user_prompt_template = """Term Name: {term_name}

Write the ISO 704:2022 definition now."""

    # Log all prompts at the top of the log file
    logger.info("="*80)
    logger.info("PROMPTS USED IN THIS RUN")
    logger.info("="*80)
    logger.info("\n--- AUDIT PROMPT 1 (alignment, run once per name/def pair) ---")
    logger.info(_ALIGNMENT_SYSTEM_PROMPT)
    logger.info("\n--- AUDIT PROMPT 2 (name distinctiveness) ---")
    logger.info(_NAMES_DISTINCT_SYSTEM_PROMPT)
    logger.info("\n--- AUDIT PROMPT 3 (definition distinction, conditional) ---")
    logger.info(_DEFS_EXPLAIN_SYSTEM_PROMPT)
    logger.info("\n--- ISO SYSTEM PROMPT ---")
    logger.info(iso_system_prompt)
    logger.info("\n--- ISO USER PROMPT TEMPLATE ---")
    logger.info(iso_user_prompt_template)
    logger.info("="*80)
    logger.info("\n")

    # region Configuration
    config = config_from_env()

    logger.info("Using Ollama")
    logger.info(f"Model: {config.model}")
    logger.info(f"Mode: {args.mode}\n")
    # endregion

    # region Client init
    try:
        client = OllamaClient(config)
    except Exception as e:
        logger.error(f"Failed to initialize Ollama client: {e}")
        logger.error("\nMake sure:")
        logger.error("1. Ollama is running: ollama serve")
        logger.error(f"2. Model is pulled: ollama pull {config.model}")
        return []
    # endregion

    # region Data loading
    if args.mode == "test":
        # region Test samples
        test_samples = [
            [
                "Push button panel door communication",
                "electrical device for activating an audible signal device to the call, which is attached on a string",
                "Hospital bell",
                "electrical device for activating an audible signal device to the call, which is attached to a string",
            ],
            [
                "Concealer stick (makeup)",
                "Concealer - as stick - is a type of cosmetic that is used to mask dark circles, age spots, large pores, and other small blemishes visible on the skin",
                "Concealer (BB cream)",
                "Concealer is a type of cosmetic that is used to mask dark circles, age spots, large pores, and other small blemishes visible on the skin",
            ],
            [
                "Bend cover for cable support system",
                "Preform which is mounted on the angular extension of a cable support system for the protection of laid cables against dust, dirt and liquids and outdoors against rain and sun",
                "Cover for corner add-on piece for cable support system",
                "Preform which is mounted on the angular extension of a cable support system for the protection of the laid cables against dust, dirt and liquids, and outdoors against rain and sun",
            ],
            [
                "Electro hydraulic actuator, self contained actuator (hydraulics)",
                "actuator that provides linear motion, with control unit, with or without motor-pump unit",
                "Cylinder, electro hydraulic actuator, self contained actuator (hydraulics)",
                "actuator that provides linear motion, with or without control unit, with or without motor-pump unit",
            ],
            [
                "Key board",
                "Key board is a storage place for keys. In public buildings, but also in apartment buildings, such a lockable box is often placed in front of or next to doors",
                "Key board, key box (living, not specified)",
                "Key board or box is a storage place for keys. In public buildings, but also in apartment buildings, such a lockable box is often placed in front of or next to doors",
            ],
            [
                "Safety-related shut-down mat (tactile sensor)",
                "Safety-related shut-down mats are often used to secure danger zones on punching systems, pipe bending machines and woodworking machines. Complete work areas are monitored",
                "Shut-down mat (tactile sensor)",
                "Shut-down mats are often used to secure danger zones on punching systems, pipe bending machines and woodworking machines. Complete work areas are monitored",
            ],
            [
                "Network Hub (general application)",
                "working exclusively on level 1 of the OSI reference model node in the network technology , which forwards the bits/symbols to all connected network participants",
                "Repeater Hub (wired, industrial application)",
                "working exclusively on level 1 of the OSI reference model node in the network technology, which forwards the bits/symbols to all connected network participants",
            ],
            [
                "Coffee cup",
                "item specially designed for coffee enjoyment",
                "Coffee mug",
                "item specially designed for the enjoyment of coffee",
            ],
            [
                "ECG/respiration cable (monitoring)",
                "cable for transmitting electrical signals of the heart and respiration from the patient to the monitoring system",
                "ECG leadwire set (monitoring)",
                "cable for transmitting electrical signals of the heart and respiration or other vital parameters from the patient to the monitoring system",
            ],
            [
                "Electrically operated valve (pneumatics)",
                "valve that opens or blocks one or more current paths (flow paths) with electrical actuation and non-standardized connection surface",
                "Standard valve electrically operated (pneumatics)",
                "valve that opens or blocks one or more current paths (flow paths) with electrical actuation and standardized connection area",
            ],
        ]
        # endregion

        # Process test samples
        results: List[Dict] = []
        for i, sample in enumerate(test_samples):
            result = analyze_tuple(
                client, sample, i,
                iso_system_prompt, iso_user_prompt_template,
                logger
            )
            results.append(result)

    else:  # real mode
        # Load data from CSV files
        data_dir = script_dir.parent / "data"
        csv_files = [
            data_dir / "4-experiment" / "classes-similarity-threshold" / "similarity_matches_thresh_0.03.csv",
            data_dir / "4-experiment" / "properties-similarity-threshold" / "similarity_matches_thresh_0.03.csv",
        ]

        all_csv_data = []
        for csv_file in csv_files:
            if csv_file.exists():
                csv_data = load_csv_data(csv_file, logger)
                all_csv_data.extend(csv_data)
            else:
                logger.warning(f"CSV file not found: {csv_file}")

        logger.info(f"Total rows loaded: {len(all_csv_data)}")

        # Process real data
        results_for_csv = []
        processed = 0
        skipped = 0

        for i, (distance, pref_name_a, text_a, pref_name_b, text_b) in enumerate(all_csv_data):
            # Skip if either definition is chemical
            if is_chemical_definition(text_a) or is_chemical_definition(text_b):
                logger.info(f"Skipping row {i}: chemical definition detected")
                skipped += 1
                continue

            # Remove IDs from preferred names for prompts
            clean_name_a = remove_id_from_preferred_name(pref_name_a)
            clean_name_b = remove_id_from_preferred_name(pref_name_b)

            # Analyze tuple
            tuple_data = [clean_name_a, text_a, clean_name_b, text_b]
            result = analyze_tuple(
                client, tuple_data, i,
                iso_system_prompt, iso_user_prompt_template,
                logger
            )

            # Extract verdict
            verdict = extract_verdict(result.get("audit_response", ""))

            # Get proposed definitions
            proposed_defs = result.get("proposed_definitions", {})
            llm_def_a = proposed_defs.get("A", "")
            llm_def_b = proposed_defs.get("B", "")

            # Create row for CSV (with original preferred names including IDs)
            csv_row = {
                "distance": distance,
                "preferred_names_a": pref_name_a,
                "text_a": text_a,
                "preferred_names_b": pref_name_b,
                "text_b": text_b,
                "verdict": verdict,
                "llm_definition_a": llm_def_a,
                "llm_definition_b": llm_def_b,
            }
            results_for_csv.append(csv_row)
            processed += 1

        logger.info(f"\nProcessed: {processed} rows")
        logger.info(f"Skipped (chemical): {skipped} rows")

        # Save results to CSV
        output_csv = data_dir / "5-experiment" / "llm_analysis.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        save_results_to_csv(results_for_csv, output_csv, logger)

        results = []  # Return empty for real mode as we save to CSV
    # endregion

    # region Summary for test mode
    if args.mode == "test":
        successful = sum(1 for r in results if r["status"] == "success")
        with_remediations = sum(
            1
            for r in results
            if r["status"] == "success" and r.get("proposed_definitions")
        )

        logger.info(
            f"\nProcessed {len(results)} tuples: {successful} successful, "
            f"{len(results) - successful} errors"
        )
        logger.info(f"New definitions proposed: {with_remediations}")
    # endregion

    return results


if __name__ == "__main__":
    main()