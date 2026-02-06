"""
Experiment 5: Analyze name-definition tuple pairs using Llama 3.1 via Ollama.

This module helps to identify inconsistencies in definitions by comparing tuple pairs.
"""

import os
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import icontract

from openai import OpenAI


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


def _needs_remediation(diagnosis: str) -> bool:
    """Determine if definitions need remediation based on diagnosis category.

    :param diagnosis: The diagnostic text from initial analysis
    :return: True if remediation is needed, False otherwise
    """
    # Categories that indicate definitions need improvement
    problematic_categories = [
        "DEFINITION INSUFFICIENT",
        "MISALIGNMENT",
    ]
    return any(category in diagnosis for category in problematic_categories)


def _parse_fix_section(diagnosis: str) -> List[str]:
    """Parse the Fix section to identify which definitions need updating.

    :param diagnosis: The complete diagnostic text
    :return: List containing 'A', 'B', both, or empty list
    """
    terms_to_fix = []  # type: List[str]

    # Find the Fix section
    lines = diagnosis.split("\n")
    fix_line = ""

    for line in lines:
        if line.strip().startswith("- Fix:") or line.strip().startswith("Fix:"):
            fix_line = line.strip()
            break

    if not fix_line:
        return terms_to_fix

    # Parse which definitions are mentioned
    fix_upper = fix_line.upper()

    if "DEF A" in fix_upper or "DEFINITION A" in fix_upper:
        terms_to_fix.append("A")

    if "DEF B" in fix_upper or "DEFINITION B" in fix_upper:
        terms_to_fix.append("B")

    return terms_to_fix


def generate_iso_definition(client: LLMClient, term_name: str) -> str:
    """Generate an ISO 704:2022 compliant definition for a term.

    :param client: LLM client to use for generation
    :param term_name: The name of the term to define
    :return: Generated definition text
    """
    system_prompt = """You are a strict Terminologist. 
Your sole purpose is to generate a single, formal definition for a provided Term Name that complies with ISO 704:2022 standards.

### Rules
1. **Format:** Output ONLY the definition text. Do not output the name, labels, or explanation.
2. **Structure:** Use the intensional definition structure: [Genus/Superordinate Concept] + [Differentia/Distinguishing Characteristics].
3. **Constraints:**
   - No circularity (do not use the term itself).
   - No encyclopedic info or examples.
   - Must be a single, complete sentence."""

    user_prompt = f"""Term Name: {term_name}

Write the ISO 704:2022 definition now."""

    definition = client.create_completion(system_prompt, user_prompt)
    return definition.strip()


def remediate_definitions(
    client: LLMClient,
    name_1: str,
    def_1: str,
    name_2: str,
    def_2: str,
    audit_response: str,
) -> Dict[str, str]:
    """Generate improved definitions based on identified issues.

    :param client: LLM client to use for generation
    :param name_1: First name
    :param def_1: First definition
    :param name_2: Second name
    :param def_2: Second definition
    :param audit_response: The complete audit diagnosis
    :return: Dictionary mapping 'A' and/or 'B' to new definitions
    """
    # Identify which definitions need fixing
    terms_to_fix = _parse_fix_section(audit_response)

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

        print(f"  Generating ISO 704 definition for '{term_name}'...")

        try:
            new_def = generate_iso_definition(client, term_name)
            new_definitions[term] = new_def
            print(f"---> {term_label}: {new_def}")
        except Exception as e:
            print(f"---> Failed to generate {term_label}: {e}")

    return new_definitions


def analyze_tuple(
    client: LLMClient,
    tuple_data: List[str],
    tuple_id: int,
    system_prompt: str,
    enable_remediation: bool = True,
) -> Dict:
    """Analyze a single name-definition pair.

    :param client: LLM client to use for analysis
    :param tuple_data: List containing [name_1, def_1, name_2, def_2]
    :param tuple_id: Number of this tuple
    :param system_prompt: System instruction for the LLM
    :param enable_remediation: Whether to generate improved definitions
    :return: Dictionary with analysis results and proposed fixes
    :raise ValueError: If tuple_data does not contain exactly 4 elements
    """
    if len(tuple_data) != 4:
        raise ValueError(
            f"Tuple {tuple_id} has {len(tuple_data)} elements, expected 4"
        )

    name_1, def_1, name_2, def_2 = tuple_data

    print(f"Tuple number {tuple_id} will be assessed:")
    print(f"\tName 1: {name_1}")
    print(f"\tDef 1: {def_1}")
    print(f"\tName 2: {name_2}")
    print(f"\tDef 2: {def_2}")
    print()

    user_prompt = f"""Input Data:
Tuple A:
- Name: {name_1}
- Definition: {def_1}

Tuple B:
- Name: {name_2}
- Definition: {def_2}

Task Instructions:
Analyze the relationship between Names and Definitions through three specific checks.

1. Internal Alignment Check
- Does Def A accurately describe Name A?
- Does Def B accurately describe Name B?

2. Distinction Gap Analysis
- Name Delta: How distinct are the two Preferred Names conceptually?
- Definition Delta: How distinct are the two Definitions?
- Consistency: Does the difference between the Definitions match the difference between the Names? (e.g., If names are distinct, do definitions explain why?)

3. Diagnostic Verdict
Choose exactly ONE audit_response from the list below:
- VALID DISTINCTION: Names differ and definitions clearly explain that difference.
- DEFINITION INSUFFICIENT: Names are distinct, but definitions are too similar to explain the distinction.
- MISALIGNMENT: One or both definitions do not accurately reflect their own Preferred Name.
- TRUE REDUNDANCY: Names are synonyms and definitions are identical.

Output Format:
### Analysis
- Alignment: [Pass/Fail check on whether definitions fit their names]
- Gap Consistency: [Does the definition difference scale with the name difference?]

### Diagnosis
- Category: [One of the 4 categories above]
- Fix: [Briefly state what needs to change (e.g., "Sharpen Def A to specify X," or "None needed")]
"""

    try:
        audit_response = client.create_completion(system_prompt, user_prompt)

        result = {
            "tuple_id": tuple_id,
            "name_1": name_1,
            "name_2": name_2,
            "audit_response": audit_response,
            "status": "success",
        }

        print(result["audit_response"])
        print("\n" + "=" * 80 + "\n")

        # region Remediation
        if enable_remediation and _needs_remediation(audit_response):
            print("\n--- Remediation Part ---")

            try:
                new_definitions = remediate_definitions(
                    client=client,
                    name_1=name_1,
                    def_1=def_1,
                    name_2=name_2,
                    def_2=def_2,
                    audit_response=audit_response,
                )

                if new_definitions:
                    print()
                    result["proposed_definitions"] = new_definitions
                else:
                    print("  No specific definitions identified for remediation.")
                    result["proposed_definitions"] = {}

            except Exception as e:
                print(f"  Remediation failed: {e}")
                result["proposed_definitions"] = {}
        else:
            result["proposed_definitions"] = {}
        # endregion

        print("\n" + "=" * 15 + "\n")

        return result

    except Exception as e:
        print(f"Error processing tuple {tuple_id}: {e}")
        return {
            "tuple_id": tuple_id,
            "name_1": name_1,
            "name_2": name_2,
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


def main() -> List[Dict]:
    """Execute the main analysis workflow.

    :return: List of analysis results for all tuples
    """
    # region Configuration
    config = config_from_env()

    print("Using Ollama")
    print(f"Model: {config.model}\n")
    # endregion

    # region Client init
    try:
        client = OllamaClient(config)
    except Exception as e:
        print(f"Failed to initialize Ollama client: {e}")
        print("\nMake sure:")
        print("1. Ollama is running: ollama serve")
        print(f"2. Model is pulled: ollama pull {config.model}")
        return []
    # endregion

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

    # region System prompt
    system_prompt = """You are a precise Terminologist and Data Quality Auditor. 
Your goal is NOT to decide if concepts should be merged, but to diagnose if the definitions accurately distinguish the provided Preferred Names. 
You must be concise (max 1-2 sentences per point) and objective."""
    # endregion

    # region Analysis execution
    results: List[Dict] = []
    for i, sample in enumerate(test_samples):
        result = analyze_tuple(client, sample, i, system_prompt)
        results.append(result)
    # endregion

    # region Summary
    successful = sum(1 for r in results if r["status"] == "success")
    with_remediations = sum(
        1
        for r in results
        if r["status"] == "success" and r.get("proposed_definitions")
    )

    print(
        f"\nProcessed {len(results)} tuples: {successful} successful, "
        f"{len(results) - successful} errors"
    )
    print(f"New definitions proposed: {with_remediations}")
    # endregion

    return results


if __name__ == "__main__":
    main()