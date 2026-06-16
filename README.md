# Serbian Paragraph-Context Hate Speech Detection

The main methodology classifies one fixed sentence at a time while giving the LLM the full paragraph as context. Sentence boundaries and IDs come from the dataset, so predictions stay aligned with sentence-level ground truth.

The preparation step joins the paragraph and sentence Excel files by `ID`. It never splits paragraph text.

## Quick Start

```bash
python prepare_paragraph_context_dataset.py
python paragraph_context_run.py
python paragraph_context_statistics.py
```

Paths are fixed relative to the repository. All models are loaded from `models/models.json`. Use `--debug 10` for a ten-paragraph test run; when omitted, all paragraphs are processed. `--debut 10` is accepted as an alias. Use `--save-raw` to add the `RawOutputs` Excel sheet.

Inference writes predictions and basic metrics to `results/paragraph_context_comparison.xlsx`. After inference is complete, run `paragraph_context_statistics.py` to create `results/paragraph_context_statistics.xlsx` with standard deviations, paragraph-bootstrap 95% confidence intervals, exact paired McNemar tests with Holm correction, binary precision/recall/F1 and balanced accuracy, category macro-F1, and runtime variability. This script never calls the LLM.

The optional ensemble votes by fixed sentence ID:

```bash
python paragraph_context_run_ensemble.py --debug 10
```

See [docs/paragraph_context_methodology.md](docs/paragraph_context_methodology.md) for the dataset format, evaluation details, and comparison with the old paragraph-splitting baseline. Existing `full_text_run.py` and `single_sentence_run.py` scripts are retained as baselines.
