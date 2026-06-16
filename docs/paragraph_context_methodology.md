# Paragraph Context Methodology

## Motivation

The old paragraph baseline sends only a paragraph to the LLM. The model first splits that paragraph and then classifies its own generated sentence list. A model can split, merge, reorder, paraphrase, or omit text, so its output cannot be aligned reliably with sentence-level ground truth.

The new methodology sends one fixed sentence per LLM call, together with its paragraph context:

1. The complete paragraph, used only as context.
2. One fixed numbered sentence, such as 1 or 2, used as the classification target for that call.

The model returns only one category code for that fixed sentence ID, for example 0, 1, 1a, or 6a. The runner normalizes those sentence-level calls back into paragraph records, and evaluation joins by ID, never by generated sentence text.

## Dataset

`data/paragraph_context_dataset.xlsx` is created by joining the paragraph and sentence workbooks on `ID`. It has these columns:

- `ID`: paragraph identifier shared by both source workbooks.
- `Text`: full paragraph.
- `Sentences`: fixed sentences separated by `|||`, allowing embedded newlines inside a sentence. Legacy newline-separated cells are also readable.
- `Category`: comma-separated sentence entries aligned with `Sentences`. Parentheses group multiple labels for one sentence, for example `(1a;6b), 0, 6c`.

Sentence text and labels come directly from `data/single_sentence_hate_speech_no_offenses.xlsx`. Paragraph context comes from `data/paragraph_hate_speech_no_offenses.xlsx`. No regex or LLM sentence splitting is performed during preparation or inference; missing sentence data is an error.

## Commands

```bash
python prepare_paragraph_context_dataset.py
python paragraph_context_run.py
python paragraph_context_run.py --debug 10 --save-raw
python paragraph_context_statistics.py
python paragraph_context_run_ensemble.py --debug 10
```

Input/output paths are hardcoded relative to the repository, and every model in `models/models.json` is used. Omitting `--debug` processes all paragraphs. `--debut` is accepted as an alias for `--debug`.

The inference run writes `Metrics` and `PerSentence`; `RawOutputs` is included only with `--save-raw`. Statistics are deliberately calculated afterward by `paragraph_context_statistics.py`, which reads the completed `PerSentence` sheet without calling any models. It writes a separate workbook containing `Summary`, `Statistics`, and `McNemar` sheets.

The old `full_text_run.py` and sentence baselines remain available for comparison.
