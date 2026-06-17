# Serbian Sentence-CoT Hate Speech Detection

The main methodology classifies one fixed sentence at a time with a sentence-only CoT prompt. Sentence boundaries and IDs come from the dataset, so predictions stay aligned with sentence-level ground truth.

The preparation step joins the paragraph and sentence Excel files by `ID`. It never splits paragraph text.

## Quick Start

```bash
python prepare_sentence_cot_dataset.py
python sentence_cot_run.py
python sentence_cot_statistics.py
```

Paths are fixed relative to the repository. All models are loaded from `models/models.json`. Use `--debug 10` for a ten-paragraph test run; when omitted, all paragraphs are processed. `--debut 10` is accepted as an alias. Use `--save-raw` to add the `RawOutputs` Excel sheet.

Inference writes predictions and basic metrics to `results/sentence_cot_comparison.xlsx`. After inference is complete, run `sentence_cot_statistics.py` to create `results/sentence_cot_statistics.xlsx` with paragraph-bootstrap 95% confidence intervals, exact paired McNemar tests with Holm correction, binary precision/recall/F1, false positive/negative rates and balanced accuracy, category macro-F1. This script never calls the LLM.

The optional ensemble votes by fixed sentence position:

```bash
python sentence_cot_ensemble.py --debug 10
```

See [docs/sentence_cot_methodology.md](docs/sentence_cot_methodology.md) for the dataset format, evaluation details, and comparison with the old paragraph-splitting baseline. Existing `full_text_run.py` and `single_sentence_run.py` scripts are retained as baselines.
