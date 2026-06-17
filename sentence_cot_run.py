"""Evaluate models on fixed sentences with sentence-only CoT."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.categories import get_category_prompt
from src.llm_detector import LLMDetector
from src.utils import build_model_tags, load_sentence_cot_dataset, parse_category_and_subcategory

REPO_ROOT = Path(__file__).resolve().parent
DATASET_PATH = REPO_ROOT / "data" / "sentence_cot_dataset.xlsx"
MODELS_PATH = REPO_ROOT / "models" / "models.json"
OUTPUT_PATH = REPO_ROOT / "results" / "sentence_cot_comparison.xlsx"


def print_result_status(
    model_name: str, paragraph_id: int, total: int, expected_sentences: int, result: dict
) -> None:
    """Print concise progress and useful diagnostics for one paragraph worth of sentence calls."""
    parse_error = str(result.get("parse_error", "") or "")
    sentence_errors = result.get("sentence_parse_errors", []) or []
    predictions = result.get("sentences", [])
    elapsed = float(result.get("elapsed_seconds", 0.0) or 0.0)
    if parse_error:
        raw_preview = " ".join(str(result.get("raw", "")).split())[:300]
        print(f"[{model_name}] paragraph {paragraph_id}/{total}: PARSE ERROR ({elapsed:.2f}s)")
        print(f"  error: {parse_error}")
        print(f"  raw:   {raw_preview or '<empty response>'}")
        return

    returned = len(predictions) if isinstance(predictions, list) else 0
    if sentence_errors:
        status = "PARTIAL PARSE ERROR"
    else:
        status = "OK" if returned == expected_sentences else "ID COUNT WARNING"
    print(
        f"[{model_name}] paragraph {paragraph_id}/{total}: {status} "
        f"({returned}/{expected_sentences} sentences, {elapsed:.2f}s)"
    )
    for item in sentence_errors[:3]:
        print(f"  {item.get('id')}: {item.get('parse_error')}")
    if len(sentence_errors) > 3:
        print(f"  ... {len(sentence_errors) - 3} more sentence parse errors")


def _primary(codes: list[str]) -> tuple[int, str, str]:
    code = next((value for value in codes if value != "0"), "0")
    parsed = parse_category_and_subcategory(code)
    return int(parsed["category"]), str(parsed["subcategory"]), code


def _prediction_for_id(predictions: list[dict], sentence_id: int) -> dict | None:
    return next((item for item in predictions if item.get("id") == sentence_id), None)


def metric_row(model: str, rows: list[dict], elapsed: float, calls: int) -> dict:
    valid_rows = [row for row in rows if row["prediction_valid"]]
    total = len(valid_rows)
    ratio = lambda count, denominator: count / denominator if denominator else None
    return {
        "model": model,
        "invalid_prediction_rate": (len(rows) - total) / len(rows) if rows else None,
        "total_seconds": elapsed,
        "average_seconds_per_llm_call": elapsed / calls if calls else None,
        "average_seconds_per_sentence": elapsed / len(rows) if rows else None,
        "binary_accuracy": ratio(sum(r["binary_correct"] for r in valid_rows), total),
        "category_accuracy": ratio(sum(r["category_correct"] for r in valid_rows), total),
        "subcategory_full_code_accuracy": ratio(sum(r["full_code_correct"] for r in valid_rows), total),
        "binary_false_positive_rate": ratio(sum((r["gt_code"] == "0") and (r["pred_code"] != "0") for r in valid_rows), sum(r["gt_code"] == "0" for r in valid_rows)),
        "binary_false_negative_rate": ratio(sum((r["gt_code"] != "0") and (r["pred_code"] == "0") for r in valid_rows), sum(r["gt_code"] != "0" for r in valid_rows)),
    }


def evaluate_predictions(model: str, records: list[dict], group_predictions: list[dict]) -> list[dict]:
    rows = []
    for group_index, (record, result) in enumerate(zip(records, group_predictions), start=1):
        paragraph_id = record.get("id", group_index)
        predictions = result.get("sentences", []) if isinstance(result, dict) else []
        sentence_errors = {
            item.get("id"): item.get("parse_error", "")
            for item in result.get("sentence_parse_errors", [])
            if isinstance(item, dict)
        } if isinstance(result, dict) else {}
        for sentence_id, (sentence, gt_codes) in enumerate(zip(record["sentences"], record["gt_codes"]), start=1):
            pred = _prediction_for_id(predictions, sentence_id)
            sentence_parse_error = sentence_errors.get(sentence_id, "")
            gt_cat, _, gt_primary = _primary(gt_codes)
            gt_has_hate = any(code != "0" for code in gt_codes)
            prediction_valid = pred is not None and not result.get("parse_error") and not sentence_parse_error
            if prediction_valid:
                pred_codes = pred.get("codes") or ["0"]
                pred_cat, _, pred_primary = _primary(pred_codes)
                pred_has_hate = bool(pred.get("has_hate_speech", pred_cat != 0))
                binary_correct = gt_has_hate == pred_has_hate
                category_correct = gt_cat == pred_cat
                full_code_correct = gt_primary == pred_primary
            else:
                pred_primary = None
                binary_correct = None
                category_correct = None
                full_code_correct = None
            rows.append({
                "model": model,
                "paragraph_id": paragraph_id,
                "sentence_id": sentence_id,
                "sentence": sentence,
                "prediction_valid": prediction_valid,
                "gt_code": gt_primary,
                "pred_code": pred_primary if prediction_valid else None,
                "binary_correct": binary_correct,
                "category_correct": category_correct,
                "full_code_correct": full_code_correct,
                "parse_error": (result.get("parse_error", "") or sentence_parse_error) if isinstance(result, dict) else "invalid result",
            })
    return rows


def run(debug: int | None = None, save_raw: bool = False) -> None:
    records = load_sentence_cot_dataset(str(DATASET_PATH))
    if debug is not None and debug > 0:
        records = records[:debug]
    categories_prompt = get_category_prompt()
    metric_rows, sentence_rows, raw_rows = [], [], []

    for model_name, model_tag in build_model_tags(models_json_path=str(MODELS_PATH)).items():
        print("\n" + "=" * 70)
        print(f"Model: {model_name} ({model_tag})")
        print(f"Paragraphs: {len(records)}")
        print("=" * 70)
        detector = LLMDetector(model_tag)
        results = []
        parse_errors = 0
        count_warnings = 0
        started = time.perf_counter()
        for paragraph_id, record in enumerate(records, start=1):
            call_started = time.perf_counter()
            result = detector.classify_sentences_cot(
                record["sentences"], categories_prompt
            )
            result["elapsed_seconds"] = time.perf_counter() - call_started
            results.append(result)
            print_result_status(
                model_name, paragraph_id, len(records), len(record["sentences"]), result
            )
            sentence_errors = result.get("sentence_parse_errors", []) or []
            if result.get("parse_error"):
                parse_errors += 1
            else:
                parse_errors += len(sentence_errors)
            if not result.get("parse_error") and len(result.get("sentences", [])) != len(record["sentences"]):
                count_warnings += 1
            if save_raw:
                raw_rows.append({
                    "model": model_name,
                    "paragraph_id": record.get("id", paragraph_id),
                    "elapsed_seconds": result["elapsed_seconds"],
                    "raw": result.get("raw", ""),
                    "parse_error": result.get("parse_error", "") or json.dumps(result.get("sentence_parse_errors", []), ensure_ascii=False),
                })
        elapsed = time.perf_counter() - started
        model_sentence_rows = evaluate_predictions(model_name, records, results)
        sentence_rows.extend(model_sentence_rows)
        llm_calls = sum(len(record["sentences"]) for record in records)
        metric_rows.append(metric_row(model_name, model_sentence_rows, elapsed, llm_calls))
        print(
            f"Summary {model_name}: {len(records)} paragraphs, "
            f"parse errors={parse_errors}, ID count warnings={count_warnings}, "
            f"total={elapsed:.2f}s, "
            f"avg/paragraph={elapsed / len(records) if records else 0.0:.2f}s, "
            f"avg/sentence-call={elapsed / llm_calls if llm_calls else 0.0:.2f}s"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        pd.DataFrame(metric_rows).to_excel(writer, index=False, sheet_name="Metrics")
        pd.DataFrame(sentence_rows).to_excel(writer, index=False, sheet_name="PerSentence")
        if save_raw:
            pd.DataFrame(raw_rows).to_excel(writer, index=False, sheet_name="RawOutputs")
    print(f"Results saved to: {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", "--debut", dest="debug", type=int, default=None,
        help="Number of paragraphs to process for testing; omitted means all paragraphs",
    )
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()
    run(args.debug, args.save_raw)


if __name__ == "__main__":
    main()
