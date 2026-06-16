"""Prepare the fixed-sentence paragraph-context dataset from legacy workbooks."""

from pathlib import Path

import pandas as pd

from src.utils import parse_gt_codes_for_sentence

REPO_ROOT = Path(__file__).resolve().parent
PARAGRAPHS_PATH = REPO_ROOT / "data" / "paragraph_hate_speech_no_offenses.xlsx"
SENTENCES_PATH = REPO_ROOT / "data" / "single_sentence_hate_speech_no_offenses.xlsx"
OUTPUT_PATH = REPO_ROOT / "data" / "paragraph_context_dataset.xlsx"


def _format_sentence_category(value) -> str:
    raw = str(value or "").strip().lower()
    parts = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    codes = []
    for part in parts:
        codes.extend(parse_gt_codes_for_sentence(part))
    unique_codes = list(dict.fromkeys(codes or ["0"]))
    return unique_codes[0] if len(unique_codes) == 1 else f"({';'.join(unique_codes)})"


def prepare(paragraphs_path: str, sentences_path: str, output: str) -> None:
    paragraphs = pd.read_excel(paragraphs_path)
    sentences = pd.read_excel(sentences_path)
    required = {"ID", "Text", "Category"}
    for path, frame in ((paragraphs_path, paragraphs), (sentences_path, sentences)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    sentence_groups = {paragraph_id: group for paragraph_id, group in sentences.groupby("ID", sort=False)}
    rows = []
    for row_index, paragraph in paragraphs.iterrows():
        paragraph_id = paragraph["ID"]
        group = sentence_groups.get(paragraph_id)
        if group is None or group.empty:
            raise ValueError(f"Paragraph ID {paragraph_id} has no rows in {sentences_path}")
        fixed_sentences = [str(value).strip() for value in group["Text"]]
        labels = [_format_sentence_category(value) for value in group["Category"]]
        if any(not sentence for sentence in fixed_sentences):
            raise ValueError(f"Paragraph ID {paragraph_id} contains an empty fixed sentence")
        rows.append({
            "ID": paragraph_id,
            "Text": str(paragraph["Text"]).strip(),
            "Sentences": " ||| ".join(fixed_sentences),
            "Category": ",".join(labels),
        })

    paragraph_ids = set(paragraphs["ID"])
    extra_sentence_ids = set(sentences["ID"]) - paragraph_ids
    if extra_sentence_ids:
        raise ValueError(f"Sentence workbook contains unknown paragraph IDs: {sorted(extra_sentence_ids)[:10]}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["ID", "Text", "Sentences", "Category"]).to_excel(
        output_path, index=False, sheet_name="ParagraphContext"
    )
    print(f"Joined {len(rows)} paragraphs with {len(sentences)} fixed sentences: {output_path}")


def main() -> None:
    prepare(str(PARAGRAPHS_PATH), str(SENTENCES_PATH), str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
