"""Run llm-verification-of-eclass as Python module."""

import llm_verification_of_eclass.main

if __name__ == "__main__":
    # The ``prog`` needs to be set in the argparse.
    # Otherwise, the program name in the help shown to the user will be ``__main__``.
    llm_verification_of_eclass.main.main(prog="llm_verification_of_eclass")
