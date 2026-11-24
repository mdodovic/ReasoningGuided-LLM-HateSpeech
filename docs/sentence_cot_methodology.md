# Sentence-CoT Methodology

## Motivation

The old paragraph baseline sends only a paragraph to the LLM. The model first splits that paragraph and then classifies its own generated sentence list. A model can split, merge, reorder, paraphrase, or omit text, so its output cannot be aligned reliably with sentence-level ground truth.

The new methodology sends one fixed sentence per LLM call and uses a sentence-only CoT prompt. The paragraph text is retained in the dataset for grouping and evaluation, but it is not sent to the model.

The model returns only one category code for that fixed sentence, for example 0, 1, 1a, or 6a. The runner normalizes those sentence-level calls back into paragraph records, and evaluation joins by ID, never by generated sentence text.

## Dataset

`data/sentence_cot_dataset.xlsx` is created by joining the paragraph and sentence workbooks on `ID`. It has these columns:

- `ID`: paragraph identifier shared by both source workbooks.
- `Text`: full paragraph.
- `Sentences`: fixed sentences separated by `|||`, allowing embedded newlines inside a sentence. Legacy newline-separated cells are also readable.
- `Category`: comma-separated sentence entries aligned with `Sentences`. Parentheses group multiple labels for one sentence, for example `(1a;6b), 0, 6c`.

Sentence text and labels come directly from `data/single_sentence_hate_speech_no_offenses.xlsx`. Paragraph text comes from `data/paragraph_hate_speech_no_offenses.xlsx` for dataset grouping. It is not included in the classification prompt. No regex or LLM sentence splitting is performed during preparation or inference; missing sentence data is an error.

## Commands

```bash
python prepare_sentence_cot_dataset.py
python sentence_cot_run.py
python sentence_cot_run.py --debug 10 --save-raw
python sentence_cot_statistics.py
python sentence_cot_ensemble.py --debug 10
```

Input/output paths are hardcoded relative to the repository, and every model in `models/models.json` is used. Omitting `--debug` processes all paragraphs. `--debut` is accepted as an alias for `--debug`.

The inference run writes `Metrics` and `PerSentence`; `RawOutputs` is included only with `--save-raw`. Statistics are deliberately calculated afterward by `sentence_cot_statistics.py`, which reads the completed `PerSentence` sheet without calling any models. It writes a separate workbook containing `Summary`, `Statistics`, and `McNemar` sheets.

The earlier paragraph-splitting baseline (`full_text_run.py`) and the single-sentence baselines live in the [HateSpeech-LLM-Detector](https://github.com/mdodovic/HateSpeech-LLM-Detector) repository.
