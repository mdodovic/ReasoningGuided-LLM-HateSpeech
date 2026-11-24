# Reasoning-Guided LLM Prompting for Serbian Hate Speech Classification

Code, data and results for the paper:

> Janko Tufegdžić, Matija Dodović, Uroš Radenković, Marko Mišić, Dražen Drašković.
> **Improving Hate Speech Classification Using Reasoning-Guided Large Language Models.**
> ERK 2026, Portorož, Slovenia.
> PDF (accepted version): [ERK_Accepted_Improving_Hate_Speech_Classification_Using_Reasoning_Guided_Large_Language_Models.pdf](ERK_Accepted_Improving_Hate_Speech_Classification_Using_Reasoning_Guided_Large_Language_Models.pdf)

The study classifies Serbian sentences into a hate speech taxonomy of 7 categories and 14 codes. It uses ten open LLMs, all run locally through [Ollama](https://ollama.com). The prompt tells the model to work through a structured decision process internally: target group, irony, quotation, negation, counter-speech and context. The model then returns only the final category code, so every response can be parsed and scored automatically.

> **Content warning:** the data contains real hateful and offensive language in Serbian.

## Method

- **One fixed sentence per call.** Sentence boundaries, IDs and labels come from the annotated dataset. The model never splits text. Predictions are joined to the ground truth by paragraph ID and sentence ID, never by generated text. The paragraph text is used for grouping and resampling only; it is not sent to the model.
- **Output format.** A single code (`0`, `1a` … `7`). Anything else is recorded as an invalid prediction.
- **Three evaluation levels.** Binary (hate vs. no hate), main category, and full code (category and subcategory). When a sentence has several labels, the first non-zero label is the primary target.
- **Statistics.** Two-sided exact McNemar tests for all 45 model pairs at each of the three levels. Holm correction is applied jointly to all 135 tests, with significance at adjusted p < 0.05. Each pair uses only sentences that are valid for both models, so comparisons with Llama 4 use 556 sentences and all others use 557. The statistics script also writes paragraph-cluster bootstrap 95% confidence intervals.
- **Evaluation set.** The first 103 paragraphs of the corpus (IDs 1–103), 557 sentences.
- **Decoding.** Temperature 0.1, at most 16 output tokens, Ollama `think` disabled.

## Results

From Tables 1 and 2 of the paper (557 sentences per model):

| Model | Valid [%] | Binary acc. | Binary F1 | Category acc. | Full-code acc. | Time / sentence [s] |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 | 100.00 | 0.732 | 0.613 | 0.585 | 0.557 | 0.12 |
| Qwen3.5 | 100.00 | 0.713 | 0.470 | **0.641** | **0.634** | 0.42 |
| Qwen3.6 | 100.00 | 0.745 | 0.587 | 0.630 | 0.614 | 4.09 |
| Mistral | 100.00 | 0.713 | 0.630 | 0.542 | 0.499 | 0.06 |
| Phi-3 | 100.00 | 0.499 | 0.508 | 0.287 | 0.255 | 0.04 |
| Phi-4 | 100.00 | **0.754** | 0.684 | 0.609 | 0.571 | 0.10 |
| Llama 3 | 100.00 | 0.461 | 0.586 | 0.144 | 0.117 | 0.12 |
| Llama 4 | 99.82 | 0.746 | 0.686 | 0.554 | 0.493 | 4.34 |
| Gemma 4 | 100.00 | 0.734 | **0.697** | 0.582 | 0.531 | 0.20 |
| Granite 4 | 100.00 | 0.470 | 0.586 | 0.147 | 0.118 | 0.08 |

Phi-4 has the highest observed binary accuracy, Gemma 4 the highest binary F1, and Qwen3.5 the highest category and full-code accuracy.

The Holm-corrected McNemar tests (Table 3 of the paper) find 21 significant pairs out of 45 for binary detection, 30 for category classification and 36 for full-code classification:

- **Binary detection.** Phi-4 is significantly better only than Phi-3, Granite 4 and Llama 3. Its gaps to its closest competitors, Qwen3.6, Llama 4 and Gemma 4, are not significant, so the experiment does not single out one best binary model.
- **Category level.** Qwen3.5 is significantly better than Qwen3, Mistral, Phi-3, Granite 4, Llama 3 and Llama 4. It is not significantly different from Qwen3.6, Phi-4 or Gemma 4.
- **Full code.** Qwen3.5 is significantly better than every model except Qwen3.6.

Table 4 of the paper compares these results with the Serbian baselines reported in Dodović et al., IEEE Access 2026.

The per-sentence predictions are in [results/sentence_cot_comparison.xlsx](results/sentence_cot_comparison.xlsx). Precision, recall, false positive and false negative rates, bootstrap intervals and all 135 McNemar tests are in [results/sentence_cot_statistics.xlsx](results/sentence_cot_statistics.xlsx).

## Repository layout

```text
prepare_sentence_cot_dataset.py   builds data/sentence_cot_dataset.xlsx from the paragraph and sentence workbooks
sentence_cot_run.py               inference: one LLM call per fixed sentence, for every model in models/models.json
sentence_cot_statistics.py        bootstrap CIs and Holm-corrected McNemar tests (no LLM calls)
sentence_cot_ensemble.py          optional majority vote across models, per sentence ID
src/                              Ollama client, prompts, taxonomy, parsing and statistics
src/prompts/classify_sentence_cot.txt   the reasoning-guided prompt used in the paper
models/models.json                model name -> Ollama tag
data/                             corpus workbooks (see below)
results/                          outputs reported in the paper
docs/sentence_cot_methodology.md  dataset format and evaluation details
erk.tex, literature.bib, REFERENCES.txt   paper source
```

## Setup

Requirements: Python 3.10 or newer, and Ollama running on `http://localhost:11434`.

```bash
python -m pip install -r requirements.txt
ollama pull qwen3      # repeat for every tag in models/models.json
```

`models/models.json` uses `:latest` tags. These tags change over time, so a rerun can use a newer model build than the one evaluated in the paper.

## Reproducing the paper

```bash
python prepare_sentence_cot_dataset.py    # optional: data/sentence_cot_dataset.xlsx is already included
python sentence_cot_run.py --debug 103    # first 103 paragraphs = 557 sentences, all 10 models
python sentence_cot_statistics.py         # reads the finished run, writes results/sentence_cot_statistics.xlsx
```

- `--debug N` limits the run to the first N paragraphs. Without it, the whole corpus (1,351 paragraphs, 8,029 sentences) is processed.
- `--save-raw` adds a `RawOutputs` sheet with the unmodified model responses.
- A run overwrites `results/sentence_cot_comparison.xlsx`. Pass a copy to the statistics script if you want to keep the published results.

Optional ensemble (writes `results/sentence_cot_ensemble.xlsx`):

```bash
python sentence_cot_ensemble.py --debug 103
```

## Data

The corpus was introduced in Dodović et al., IEEE Access 2026 ([DOI 10.1109/ACCESS.2026.3726865](https://doi.org/10.1109/ACCESS.2026.3726865)). It has 1,351 paragraphs and 8,029 sentences from Serbian news portals and social networks. Two annotators labelled every sentence independently and a senior annotator resolved disagreements.

| File | Content |
|---|---|
| `single_sentence_hate_speech_no_offenses.xlsx` | 8,029 sentences (`ID`, `Text`, `Category`). Offensive but not hateful text is merged into `0`. Used by this study. |
| `single_sentence_hate_speech_offenses.xlsx` | The same sentences with the separate offensive label `U` kept |
| `paragraph_hate_speech_no_offenses.xlsx` | 1,351 paragraphs. Used by this study. |
| `paragraph_hate_speech_offenses.xlsx` | Paragraphs with the offensive label kept |
| `sentence_cot_dataset.xlsx` | The joined input: `ID`, `Text`, `Sentences` (separated by `\|\|\|`), `Category` (aligned labels; `(1a;6b)` marks several labels on one sentence) |

`ID` is the paragraph identifier and is shared by all files.

### Taxonomy

| Code | Category | Subcategories |
|---|---|---|
| 0 | No hate speech | – |
| 1 | Racial and ethnic/national hate | 1a race / skin color, 1b ethnicity, 1c nationality / origin |
| 2 | Religious hate | – |
| 3 | Sex- and gender-based hate | 3a sexism, 3b LGBTQ+ identities |
| 4 | Physical traits and health | 4a appearance, 4b illness / disability |
| 5 | Age- and generation-based hate | – |
| 6 | Socioeconomic hate | 6a status / class, 6b occupation, 6c political intolerance |
| 7 | Sports and fan-based hate | – |

## Related repositories

- [HateSpeech-LLM-Detector](https://github.com/mdodovic/HateSpeech-LLM-Detector): corpus and LLM benchmark (IEEE Access 2026)
- [Paragraph-HateSpeech-LLM](https://github.com/JankoTufegdzic/Paragraph-HateSpeech-LLM): the same sentences classified with paragraph context
- [Hate_Speech_with_Emojis](https://github.com/JankoTufegdzic/Hate_Speech_with_Emojis): effect of emojis on LLM hate speech detection

## Citation

```bibtex
@inproceedings{tufegdzic2026reasoning,
  author    = {Tufegd{\v{z}}i{\'c}, Janko and Dodovi{\'c}, Matija and Radenkovi{\'c}, Uro{\v{s}} and Mi{\v{s}}i{\'c}, Marko and Dra{\v{s}}kovi{\'c}, Dra{\v{z}}en},
  title     = {Improving Hate Speech Classification Using Reasoning-Guided Large Language Models},
  booktitle = {Proceedings of the International Electrotechnical and Computer Science Conference (ERK 2026)},
  address   = {Portoro{\v{z}}, Slovenia},
  year      = {2026}
}
```

## Acknowledgment

The research was partially conducted in the premises of the Palace of Science, Miodrag Kostić Endowment – Centre for Applied Artificial Intelligence. It was financially supported by the Science Fund of the Republic of Serbia, Grant No. 11113, *Software for Text Offences Prevention in Serbian: AI-driven Hate Speech Detection – STOP*, and by the Ministry of Science, Technological Development and Innovation of the Republic of Serbia, contract 451-03-34/2026-03/200103.
