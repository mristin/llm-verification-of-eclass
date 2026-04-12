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
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Final
import icontract

from openai import OpenAI

# Import LoggerFactory from the common module
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


STRUCTURAL_DEFINITIONS: Final[frozenset] = frozenset({
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
})

CHEMICAL_COMPOUNDS: Final[frozenset] = frozenset([
    "benz", "phenyl", "sulfon", "sulfonate", "sulfate", "sulfide", "sulfite", "sulfonyl",
    "nitro", "nitrate", "amino", "imino", "carbox", "carbonyl", "aminocarbonyl", "carboxyl",
    "carboxylate", "amide", "amidate", "acetate", "propionate", "butyrate", "oxalate", "lactate",
    "phosphate", "hydroxide", "oxide", "peroxide", "cyano", "oxo", "azo", "triazine", "pyridine",
    "pyrimidine", "imidazole", "quinazoline", "indole", "naphth", "anthrac", "vinyl", "vinylen",
    "ethyl", "propyl", "butyl", "methyl", "hydroxy", "methoxy", "phenoxy", "benzoyl", "benzyl",
    "ylidene", "ylide", "yl", "anilino", "naphthyl", "anthryl", "chloro", "bromo", "fluoro",
    "dichloro", "trichloro", "chloride", "chlorophenyl", "dichlorophenyl", "trifluoromethyl",
    "sodium", "potassium", "ammonium", "lithium", "magnesium", "salts", "hepta", "hexa", "penta",
    "tetra", "tri", "di", "mono", "bis", "tris", "aluminium", "paraffins", "phosphating",
    "acrylates", "hexafluoride", "arenes", "polymethyl", "maltene", "sulfate", "sulphooxy",
    "terephthalat", "phosphonat", "sulphate", "polyolefin", "onhydrochlorid"
])


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
    Returns True if at least one of the two texts is a chemical definition,
    in which case the pair should be skipped entirely.
    The check is case-insensitive.
    """
    a_lower = text_a.lower() if text_a else ""
    b_lower = text_b.lower() if text_b else ""

    if any(keyword in a_lower for keyword in CHEMICAL_COMPOUNDS):
        return True

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
    :ivar defs_too_similar: Whether the definitions are too similar to explain the name distinction.
                             True = problem found = DEFINITION INSUFFICIENT.
                             None when names_distinct is False — prompt skipped.
    :ivar defs_too_similar_justification: Justification, or None when skipped
    """

    alignment_a: bool
    alignment_b: bool
    alignment_a_justification: str
    alignment_b_justification: str
    names_distinct: bool
    names_distinct_justification: str
    defs_too_similar: Optional[bool]
    defs_too_similar_justification: Optional[str]


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


_ALIGNMENT_SYSTEM_PROMPT: Final[str] = (
    "You are a precise terminologist. "
    "Answer only YES or NO, then one sentence of justification. "
    "Format: YES/NO — [one sentence]"
)

_NAMES_DISTINCT_SYSTEM_PROMPT: Final[str] = (
    "You are a precise terminologist. "
    "Answer only YES or NO, then one sentence of justification. "
    "Format: YES/NO — [one sentence]"
)

_DEFS_TOO_SIMILAR_SYSTEM_PROMPT: Final[str] = (
    "You are a precise terminologist. "
    "Answer only YES or NO, then one sentence of justification. "
    "Answer YES if the definitions are nearly identical or refer to the exact same thing with only minor wording differences. "
    "Answer NO if the definitions clearly describe different objects, even if they share some abstract category. "
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


def _check_defs_too_similar(
        client: LLMClient,
        name_1: str, def_1: str,
        name_2: str, def_2: str,
) -> Tuple[bool, str]:
    """Prompt 3 — ask whether definitions are too similar to explain the distinction between names.

    Framed as problem detection (YES = problem exists) rather than confirmation.
    Only called when names are already confirmed distinct.

    :param client: LLM client
    :param name_1: First preferred name
    :param def_1: First definition
    :param name_2: Second preferred name
    :param def_2: Second definition
    :return: Tuple of (defs_too_similar result, justification).
             True means the definitions fail to distinguish — DEFINITION INSUFFICIENT.
    """
    user_prompt = (
        "Are these two definitions nearly identical or describing the exact same thing?\n\n"
        f"Name A: {name_1}\n"
        f"Definition A: {def_1}\n\n"
        f"Name B: {name_2}\n"
        f"Definition B: {def_2}\n\n"
        "Answer YES only if the definitions are describing the SAME specific thing with minor wording differences.\n"
        "Answer NO if the definitions clearly describe different objects with distinct functions, regardless of any broad similarities.\n\n"
        "Answer format: YES/NO — [one sentence]"
    )
    response = client.create_completion(_DEFS_TOO_SIMILAR_SYSTEM_PROMPT, user_prompt)
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
        defs_too_similar, just_defs = _check_defs_too_similar(
            client, name_1, def_1, name_2, def_2
        )
        logger.info(f"  [Prompt 3]  Defs too similar: {'YES' if defs_too_similar else 'NO'} — {just_defs}")
    else:
        defs_too_similar = None
        just_defs = None
        logger.info("  [Prompt 3]  Skipped (names not distinct)")

    return AuditSignals(
        alignment_a=alignment_a,
        alignment_b=alignment_b,
        alignment_a_justification=just_a,
        alignment_b_justification=just_b,
        names_distinct=names_distinct,
        names_distinct_justification=just_names,
        defs_too_similar=defs_too_similar,
        defs_too_similar_justification=just_defs,
    )


def compute_verdict(signals: AuditSignals) -> str:
    """Derive a deterministic verdict from the three audit signals.

    Decision table:
    - Any alignment fails                       → MISALIGNMENT
    - Both align, names not distinct            → TRUE REDUNDANCY
    - Both align, names distinct, defs similar  → DEFINITION INSUFFICIENT
    - Both align, names distinct, defs distinct → VALID DISTINCTION

    Note: defs_too_similar uses problem-detection polarity —
    True means the definitions fail to distinguish (DEFINITION INSUFFICIENT).

    :param signals: Populated AuditSignals instance
    :return: One of the four verdict strings
    """
    if not signals.alignment_a or not signals.alignment_b:
        return "MISALIGNMENT"
    if not signals.names_distinct:
        return "TRUE REDUNDANCY"
    if signals.defs_too_similar:
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
        return preferred_name[:pos + 1].strip()

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
    csv.field_size_limit(sys.maxsize)
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


_ABLATION_MODELS: Final[List[str]] = [
    "llama3.1:latest",
    "gemma2:9b",
    "qwen2.5:7b",
    "qwen3:8b",
    "olmo2:7b",
]

_TEST_TUPLES: Final[frozenset] = frozenset({
    (
        "Push button panel door communication",
        "electrical device for activating an audible signal device to the call, which is attached on a string",
        "Hospital bell",
        "electrical device for activating an audible signal device to the call, which is attached to a string",
    ),
    (
        "Concealer stick (makeup)",
        "Concealer - as stick - is a type of cosmetic that is used to mask dark circles, age spots, large pores, and other small blemishes visible on the skin",
        "Concealer (BB cream)",
        "Concealer is a type of cosmetic that is used to mask dark circles, age spots, large pores, and other small blemishes visible on the skin",
    ),
    (
        "Bend cover for cable support system",
        "Preform which is mounted on the angular extension of a cable support system for the protection of laid cables against dust, dirt and liquids and outdoors against rain and sun",
        "Cover for corner add-on piece for cable support system",
        "Preform which is mounted on the angular extension of a cable support system for the protection of the laid cables against dust, dirt and liquids, and outdoors against rain and sun",
    ),
    (
        "Electro hydraulic actuator, self contained actuator (hydraulics)",
        "actuator that provides linear motion, with control unit, with or without motor-pump unit",
        "Cylinder, electro hydraulic actuator, self contained actuator (hydraulics)",
        "actuator that provides linear motion, with or without control unit, with or without motor-pump unit",
    ),
    (
        "Key board",
        "Key board is a storage place for keys. In public buildings, but also in apartment buildings, such a lockable box is often placed in front of or next to doors",
        "Key board, key box (living, not specified)",
        "Key board or box is a storage place for keys. In public buildings, but also in apartment buildings, such a lockable box is often placed in front of or next to doors",
    ),
    (
        "Safety-related shut-down mat (tactile sensor)",
        "Safety-related shut-down mats are often used to secure danger zones on punching systems, pipe bending machines and woodworking machines. Complete work areas are monitored",
        "Shut-down mat (tactile sensor)",
        "Shut-down mats are often used to secure danger zones on punching systems, pipe bending machines and woodworking machines. Complete work areas are monitored",
    ),
    (
        "Network Hub (general application)",
        "working exclusively on level 1 of the OSI reference model node in the network technology , which forwards the bits/symbols to all connected network participants",
        "Repeater Hub (wired, industrial application)",
        "working exclusively on level 1 of the OSI reference model node in the network technology, which forwards the bits/symbols to all connected network participants",
    ),
    (
        "Coffee cup",
        "item specially designed for coffee enjoyment",
        "Coffee mug",
        "item specially designed for the enjoyment of coffee",
    ),
    (
        "ECG/respiration cable (monitoring)",
        "cable for transmitting electrical signals of the heart and respiration from the patient to the monitoring system",
        "ECG leadwire set (monitoring)",
        "cable for transmitting electrical signals of the heart and respiration or other vital parameters from the patient to the monitoring system",
    ),
    (
        "Electrically operated valve (pneumatics)",
        "valve that opens or blocks one or more current paths (flow paths) with electrical actuation and non-standardized connection surface",
        "Standard valve electrically operated (pneumatics)",
        "valve that opens or blocks one or more current paths (flow paths) with electrical actuation and standardized connection area",
    ),
})

_TEST_UNRELATED_CROSS_DOMAIN: Final[frozenset] = frozenset({
    (
        "Mobile soldering, welding device (maintenance)",
        "Portable device that is used to join or repair metal parts by soldering or welding and can be used flexibly in different locations.",
        "TV/DVD combination (plasma)",
        "Combination of a television with plasma technology and a DVD player or DVD recorder",
    ),
    (
        "TV/DVD combination (plasma)",
        "Combination of a television with plasma technology and a DVD player or DVD recorder",
        "Load safety net",
        "Means of securing loads in road, rail, air and shipping traffic against the physical forces of movement occurring during transport",
    ),
    (
        "Load safety net",
        "Means of securing loads in road, rail, air and shipping traffic against the physical forces of movement occurring during transport",
        "Tool for primary molding from plastic state",
        "tool for producing a solid body through primary shaping from the plastic state according to classification number 1.2 of DIN 8580",
    ),
    (
        "Tool for primary molding from plastic state",
        "tool for producing a solid body through primary shaping from the plastic state according to classification number 1.2 of DIN 8580",
        "Floor filler",
        "An filling compound for the floor area is suitable for the following areas: production of a necessary, uniform absorbency of the substrate when using dispersion adhesives, Preparation of a water-resistant buffer layer when using dispersion adhesives, Producing a sufficient flatness of the substrate, For repairing the ground (filling holes, etc.), Filling of slanting to adjoining components, preparation of precipitation situations, etc., as well as the use as a thin screed with elaborate restoration measures",
    ),
    (
        "Mobile soldering, welding device (maintenance)",
        "Portable device that is used to join or repair metal parts by soldering or welding and can be used flexibly in different locations.",
        "Floor filler",
        "An filling compound for the floor area is suitable for the following areas: production of a necessary, uniform absorbency of the substrate when using dispersion adhesives, Preparation of a water-resistant buffer layer when using dispersion adhesives, Producing a sufficient flatness of the substrate, For repairing the ground (filling holes, etc.), Filling of slanting to adjoining components, preparation of precipitation situations, etc., as well as the use as a thin screed with elaborate restoration measures",
    ),
    (
        "Citrus press",
        "device used for squeezing the juice out of lemons and/or oranges and/or grapefruit",
        "Axial fan",
        "Axial fans are a type of turbomachinery where the air flows parallel to the rotational axis of the impeller. Characterized by their propeller-like impeller geometry, they are suitable for applications requiring high airflow rates at relatively low pressure increases",
    ),
    (
        "Almond",
        "yellowish-white kernel of almond fruit surrounded by brown skin",
        "Swimming pool clock",
        "Digital or mechanical clock used in swimming pools, usually equipped with an unbreakable plastic glass and able to withstand humidity and similar",
    ),
    (
        "Folding bed",
        "Equivalent to the general description of a bed, but equipped with the capability of being foldable, eg for the temporary usage by guests",
        "Diaphragm valve",
        "EN 736-1: valve in which the fluid flow passage through the valve is changed by deformation of a flexible obturator.",
    ),
    (
        "Manual fire alarm",
        "Hand fire-alarm box (earlier also push button alarm unit) is a non-automatic fire alarm unit",
        "Camping grill",
        "Camping grill is a portable cooking device specifically designed for grilling food outdoors, especially while camping. It can run on various fuels such as charcoal, propane or gas",
    ),
    (
        "Warning triangle",
        "Foldable, triangular signaling device with a reflective surface that is set up behind broken-down vehicles to warn other road users.",
        "Necklace (clothing)",
        "Necklace is an article of jewellery that is worn around the neck. Necklaces may have been one of the earliest types of adornment worn by humans. They often serve ceremonial, religious, magical, or funerary purposes and are also used as symbols of wealth and status, given that they are commonly made of precious metals and stones",
    ),
})

_TEST_UNRELATED_SAME_DOMAIN: Final[frozenset] = frozenset({
    (
        "Servo cable / motor cable",
        "Cable type used to connect motors and servomotors and a control unit, whose structure and materials are specially designed for these applications",
        "Test plugs and modules (test equipment)",
        "Auxiliary equipment as test plug, test module with different switching contacts, connecting cable, applications, etc",
    ),
    (
        "Test plugs and modules (test equipment)",
        "Auxiliary equipment as test plug, test module with different switching contacts, connecting cable, applications, etc",
        "Connection component for cable floor system",
        "Component for floor integration for connecting electrical equipment",
    ),
    (
        "Connection component for cable floor system",
        "Component for floor integration for connecting electrical equipment",
        "Accessories flush mounted frame",
        "Basic element for fixation on ground on which the bonnet mounted is",
    ),
    (
        "Accessories flush mounted frame",
        "Basic element for fixation on ground on which the bonnet mounted is",
        "Signal post",
        "Beacon, usually with several different color schemes, for visual signaling of states by means of lamps",
    ),
    (
        "Servo cable / motor cable",
        "Cable type used to connect motors and servomotors and a control unit, whose structure and materials are specially designed for these applications",
        "Signal post",
        "Beacon, usually with several different color schemes, for visual signaling of states by means of lamps",
    ),
    (
        "Pressure gauge",
        "gauge that measures and indicates an absolute or a gauge pressure",
        "Bell",
        "Percussion-idiophone, thus an immediately struck sonorous resonant body (of sound) as a signaling instrument",
    ),
    (
        "USB cable",
        "Ready-made USB cable (Universal Serial Bus) is a cable used to connect, transfer data and power between electronics. USB cables have different connector types (e.g. USB-A, USB-C, ...), each of which is suitable for different applications",
        "Control cabinet system",
        "control cabinet system contains electronic equipment, parts of numerical and/or programmable logic controllers, which are required to operate, control and monitor machines or systems.",
    ),
    (
        "Starter battery",
        "starter battery is an electrical energy source that is used to operate a vehicle's starter motor and start the combustion engine.",
        "LED-lamp/Multi-LED",
        "electronic semiconductor crystal through which current flows and which emits light in the colors red, green, yellow or blue, depending on the nature of the semiconductor elements",
    ),
    (
        "Floor-type distributor",
        "Standing, large housing in which protection and switching devices are stored safely and isolated",
        "Ultrasonic motor",
        "Drive mechanism that uses high-frequency ultrasonic vibrations, generated by the piezoelectric effect, to generate movement and thus rotation in a system.",
    ),
    (
        "Gear motor for door control systems",
        "Drive unit for door control systems which represent a combination of transmission, motor and sensor",
        "Earthing set",
        "Plurality of parts for the preparation of an electrically conductive connection to the electrical potential of the ground",
    ),
})

_ABLATION_TEST_SETS: Final[Dict[str, frozenset]] = {
    "test": _TEST_TUPLES,
    "test-unrelated-cross-domain": _TEST_UNRELATED_CROSS_DOMAIN,
    "test-unrelated-same-domain": _TEST_UNRELATED_SAME_DOMAIN,
}


def save_ablation_results_to_csv(
        rows: List[Dict],
        output_path: Path,
        logger: logging.Logger,
) -> None:
    """Save ablation results to CSV file.

    :param rows: List of result row dictionaries
    :param output_path: Path to output CSV file
    :param logger: Logger instance
    """
    if not rows:
        logger.warning("No ablation results to save")
        return

    fieldnames = [
        "model", "mode", "tuple_id",
        "name_1", "name_2",
        "verdict", "status",
        "alignment_a", "alignment_a_justification",
        "alignment_b", "alignment_b_justification",
        "names_distinct", "names_distinct_justification",
        "defs_too_similar", "defs_too_similar_justification",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} ablation rows to {output_path}")


def run_ablation(
        iso_system_prompt: str,
        iso_user_prompt_template: str,
        ablation_dir: Path,
        logger: logging.Logger,
        model_filter: Optional[List[str]] = None,
        append_to: Optional[Path] = None,
) -> None:
    """Run the ablation experiment across models and test sets.

    For each combination of model × test-set, runs all audit prompts and saves
    per-combination CSV files plus a single combined CSV to ablation_dir.
    When model_filter is given, only those models are run.
    When append_to is given, existing rows from that CSV are loaded first so
    the written combined CSV contains both old and new results.

    :param iso_system_prompt: System prompt for ISO definition generation
    :param iso_user_prompt_template: User prompt template for ISO definition generation
    :param ablation_dir: Directory where ablation results are written
    :param logger: Logger instance
    :param model_filter: If given, run only these model names (must be in _ABLATION_MODELS)
    :param append_to: If given, load existing rows from this CSV before writing combined output
    """
    ablation_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Seed all_rows from existing combined CSV when appending
    all_rows: List[Dict] = []
    if append_to and append_to.exists():
        import csv as _csv
        with open(append_to, encoding="utf-8") as f:
            all_rows = list(_csv.DictReader(f))
        logger.info(f"Loaded {len(all_rows)} existing rows from {append_to}")

    models_to_run = [m for m in _ABLATION_MODELS if model_filter is None or m in model_filter]

    for model_name in models_to_run:
        safe_model = model_name.replace(":", "_").replace("/", "_")
        logger.info("=" * 80)
        logger.info(f"ABLATION — model: {model_name}")
        logger.info("=" * 80)

        config = OllamaConfig(model=model_name, temperature=0.0)
        try:
            client = OllamaClient(config)
        except Exception as e:
            logger.error(f"  Skipping model {model_name}: {e}")
            continue

        model_rows: List[Dict] = []

        for mode_name, samples in _ABLATION_TEST_SETS.items():
            logger.info(f"\n--- Test set: {mode_name} ---")

            for i, sample in enumerate(samples):
                result = analyze_tuple(
                    client, sample, i,
                    iso_system_prompt, iso_user_prompt_template,
                    logger,
                    enable_remediation=False,
                )

                signals: Optional[AuditSignals] = result.get("audit_signals")
                row: Dict = {
                    "model": model_name,
                    "mode": mode_name,
                    "tuple_id": result["tuple_id"],
                    "name_1": result["name_1"],
                    "name_2": result["name_2"],
                    "verdict": result.get("audit_response", ""),
                    "status": result["status"],
                    "alignment_a": signals.alignment_a if signals else "",
                    "alignment_a_justification": signals.alignment_a_justification if signals else "",
                    "alignment_b": signals.alignment_b if signals else "",
                    "alignment_b_justification": signals.alignment_b_justification if signals else "",
                    "names_distinct": signals.names_distinct if signals else "",
                    "names_distinct_justification": signals.names_distinct_justification if signals else "",
                    "defs_too_similar": signals.defs_too_similar if signals else "",
                    "defs_too_similar_justification": signals.defs_too_similar_justification if signals else "",
                }
                model_rows.append(row)
                all_rows.append(row)

        # Per-model CSV
        per_model_csv = ablation_dir / f"{timestamp}_{safe_model}.csv"
        save_ablation_results_to_csv(model_rows, per_model_csv, logger)

    # Combined CSV — append to existing file if requested, otherwise create new
    combined_csv = append_to if append_to else ablation_dir / f"{timestamp}_ablation_combined.csv"
    save_ablation_results_to_csv(all_rows, combined_csv, logger)

    # Summary table
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Model':<30} {'Mode':<35} {'Verdict':<25} {'Count':>5}")
    logger.info("-" * 95)

    counts: Dict[Tuple[str, str, str], int] = Counter(
        (r["model"], r["mode"], r["verdict"]) for r in all_rows if r["status"] == "success"
    )
    for (model, mode, verdict), count in sorted(counts.items()):
        logger.info(f"{model:<30} {mode:<35} {verdict:<25} {count:>5}")


def main() -> List[Dict]:
    """Execute the main analysis workflow.

    :return: List of analysis results for all tuples
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze definition tuples using LLM")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "test-unrelated-cross-domain", "test-unrelated-same-domain", "real", "ablation"],
        required=True,
        help="Mode to run: 'test' for original test samples, "
             "'test-unrelated-cross-domain' for completely unrelated concepts from different domains, "
             "'test-unrelated-same-domain' for unrelated concepts from same ECLASS domain, "
             "'real' for CSV data, "
             "'ablation' to run all 3 test sets across llama3.1, gemma2:9b, qwen2.5:7b, and qwen3:8b"
    )
    parser.add_argument(
        "--ablation-model",
        type=str,
        default=None,
        help="Comma-separated list of model names to run in ablation mode (default: all in _ABLATION_MODELS)"
    )
    parser.add_argument(
        "--append-to",
        type=str,
        default=None,
        help="Path to an existing combined ablation CSV to append new results to"
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
    # _NAMES_DISTINCT_SYSTEM_PROMPT, _DEFS_TOO_SIMILAR_SYSTEM_PROMPT) used directly
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
    logger.info("=" * 80)
    logger.info("PROMPTS USED IN THIS RUN")
    logger.info("=" * 80)
    logger.info("\n--- AUDIT PROMPT 1 (alignment, run once per name/def pair) ---")
    logger.info(_ALIGNMENT_SYSTEM_PROMPT)
    logger.info("\n--- AUDIT PROMPT 2 (name distinctiveness) ---")
    logger.info(_NAMES_DISTINCT_SYSTEM_PROMPT)
    logger.info("\n--- AUDIT PROMPT 3 (definition similarity, conditional) ---")
    logger.info(_DEFS_TOO_SIMILAR_SYSTEM_PROMPT)
    logger.info("\n--- ISO SYSTEM PROMPT ---")
    logger.info(iso_system_prompt)
    logger.info("\n--- ISO USER PROMPT TEMPLATE ---")
    logger.info(iso_user_prompt_template)
    logger.info("=" * 80)
    logger.info("\n")

    # region Ablation mode — runs before single-model setup
    if args.mode == "ablation":
        ablation_dir = experiment_dir / "ablation"
        model_filter = [m.strip() for m in args.ablation_model.split(",")] if args.ablation_model else None
        append_to = Path(args.append_to) if args.append_to else None
        active_models = model_filter if model_filter else _ABLATION_MODELS
        logger.info(f"Mode: ablation")
        logger.info(f"Models: {active_models}")
        logger.info(f"Output dir: {ablation_dir}")
        if append_to:
            logger.info(f"Appending to: {append_to}")
        logger.info("")
        run_ablation(
            iso_system_prompt, iso_user_prompt_template, ablation_dir, logger,
            model_filter=model_filter, append_to=append_to,
        )
        return []
    # endregion

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
        results: List[Dict] = []
        for i, sample in enumerate(_TEST_TUPLES):
            result = analyze_tuple(
                client, sample, i,
                iso_system_prompt, iso_user_prompt_template,
                logger
            )
            results.append(result)

    elif args.mode == "test-unrelated-cross-domain":
        logger.info("=== Testing unrelated concepts from different ECLASS domains ===\n")
        results: List[Dict] = []
        for i, sample in enumerate(_TEST_UNRELATED_CROSS_DOMAIN):
            result = analyze_tuple(
                client, sample, i,
                iso_system_prompt, iso_user_prompt_template,
                logger
            )
            results.append(result)

    elif args.mode == "test-unrelated-same-domain":
        logger.info("   Testing unrelated concepts from same ECLASS domain (27) ===\n")
        results: List[Dict] = []
        for i, sample in enumerate(_TEST_UNRELATED_SAME_DOMAIN):
            result = analyze_tuple(
                client, sample, i,
                iso_system_prompt, iso_user_prompt_template,
                logger
            )
            results.append(result)

    else:  # real mode
        # Load data from CSV files
        data_dir = script_dir.parent.parent / "data"
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
            if is_chemical_definition(text_a, text_b):
                logger.info(f"Skipping row {i}: chemical definition detected")
                skipped += 1
                continue

            # Skip if either definition is a structural placeholder
            if is_structural_definition(text_a, text_b):
                logger.info(f"Skipping row {i}: structural definition detected")
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
        logger.info(f"Skipped (chemical or structural): {skipped} rows")

        # Save results to CSV
        output_csv = data_dir / "5-experiment" / "llm_analysis.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        save_results_to_csv(results_for_csv, output_csv, logger)

        results = []  # Return empty for real mode as we save to CSV
    # endregion

    # region Summary for test modes
    if args.mode in ("test", "test-unrelated-cross-domain", "test-unrelated-same-domain"):
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

        # Show verdict breakdown
        verdict_counts = {}
        for r in results:
            if r["status"] == "success":
                verdict = r.get("audit_response", "UNKNOWN")
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        logger.info("\nVerdict breakdown:")
        for verdict, count in sorted(verdict_counts.items()):
            logger.info(f"  {verdict}: {count}")

        # For unrelated test modes, validate expected behavior
        if args.mode in ("test-unrelated-cross-domain", "test-unrelated-same-domain"):
            expected_verdict = "VALID DISTINCTION"
            unexpected = [
                (i, r.get("audit_response", "UNKNOWN"))
                for i, r in enumerate(results)
                if r["status"] == "success" and r.get("audit_response") != expected_verdict
            ]

            if unexpected:
                logger.warning(
                    f"\nWARNING: {len(unexpected)} tuples did NOT receive expected verdict '{expected_verdict}':"
                )
                for idx, verdict in unexpected:
                    logger.warning(f"  Tuple {idx}: got '{verdict}'")
            else:
                logger.info(f"\n✓ All tuples correctly identified as '{expected_verdict}'")
    # endregion

    return results


if __name__ == "__main__":
    main()