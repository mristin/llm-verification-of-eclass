# llm-verification-of-eclass

Verify the data quality of eClass with LLMs.

---

## Human Assessment

Participate in the ablation study by answering the same audit prompts the LLMs receive.
No data files and no Ollama installation are needed — the test cases are built into the script.

### Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — install with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```bash
git clone https://github.com/mristin/llm-verification-of-eclass.git
cd llm-verification-of-eclass
uv sync
```

### Run

```bash
uv run python llm_verification_of_eclass/experiments/human_assessment.py
```

You will be asked for your name at the start. Your responses are saved to
`llm_verification_of_eclass/experiments/experiment-5/ablation/<your_name>_responses.csv`
after each test set, so progress is not lost if you stop early.
