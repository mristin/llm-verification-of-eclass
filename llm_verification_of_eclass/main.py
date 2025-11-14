"""Verify the data quality of eClass with the LLMs."""

import argparse
import enum
import pathlib
import sys
from typing import TextIO

import llm_verification_of_eclass

assert llm_verification_of_eclass.__doc__ == __doc__


def main(prog: str) -> int:
    """
    Execute the main routine.

    :param prog: name of the program to be displayed in the help
    :return: exit code
    """
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--version", help="show the current version and exit", action="store_true"
    )

    # NOTE (mristin):
    # The module ``argparse`` is not flexible enough to understand special options such
    # as ``--version`` so we manually hard-wire.
    if "--version" in sys.argv and "--help" not in sys.argv:
        print(llm_verification_of_eclass.__version__)
        return 0

    args = parser.parse_args()

    return 0

def entry_point() -> int:
    """Provide an entry point for a console script."""
    return main(prog="llm-verification-of-eclass")


if __name__ == "__main__":
    sys.exit(main(prog="llm-verification-of-eclass"))
