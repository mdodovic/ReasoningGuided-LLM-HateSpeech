"""Majority-vote ensemble over fixed sentence IDs."""

import argparse
import time
from collections import Counter

import pandas as pd

from sentence_cot_run import (
    DATASET_PATH,
    MODELS_PATH,
    REPO_ROOT,
    evaluate_predictions,
    metric_row,
    print_result_status,
)
from src.categories import get_category_prompt
from src.llm_detector import LLMDetector
from src.utils import build_model_tags, load_sentence_cot_dataset

OUTPUT_PATH = REPO_ROOT / "results" / "sentence_cot_ensemble.xlsx"


def _vote(predictions: list[dict], sentence_id: int) -> dict | None:
    candidates = []
    for result in predictions:
        if result.get("parse_error"):
            continue
        item = next((value for value in result.get("sentences", []) if value.get("id") == sentence_id), None)
        if item is not None:
            candidates.append(item)
    if not candidates:
        return None
    hate = sum(bool(item.get("has_hate_speech")) for item in candidates) > len(candidates) / 2
    if not hate:
        return {"id": sentence_id, "has_hate_speech": False, "category": 0, "subcategory": "", "codes": ["0"]}
    nonzero = [code for item in candidates for code in item.get("codes", []) if code != "0"]
    code = sorted(Counter(nonzero).items(), key=lambda pair: (-pair[1], pair[0]))[0][0] if nonzero else "0"
    return {"id": sentence_id, "has_hate_speech": code != "0", "category": int(code[0]), "subcategory": code[1:] if len(code) > 1 else "", "codes": [code]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", "--debut", dest="debug", type=int, default=None,
        help="Number of paragraphs to process for testing; omitted means all paragraphs",
    )
    args = parser.parse_args()
    records = load_sentence_cot_dataset(str(DATASET_PATH))
    if args.debug is not None and args.debug > 0:
        records = records[:args.debug]
    model_tags = build_model_tags(models_json_path=str(MODELS_PATH))
    detectors = {name: LLMDetector(tag) for name, tag in model_tags.items()}
    categories_prompt = get_category_prompt()
    ensemble_results = []
    model_elapsed = {name: 0.0 for name in detectors}
    run_started = time.perf_counter()
    for paragraph_id, record in enumerate(records, start=1):
        model_results = []
        for model_name, detector in detectors.items():
            call_started = time.perf_counter()
            result = detector.classify_sentences_cot(
                record["sentences"], categories_prompt
            )
            call_elapsed = time.perf_counter() - call_started
            result["elapsed_seconds"] = call_elapsed
            model_elapsed[model_name] += call_elapsed
            print_result_status(
                model_name, paragraph_id, len(records), len(record["sentences"]), result
            )
            model_results.append(result)
        voted = [_vote(model_results, sid) for sid in range(1, len(record["sentences"]) + 1)]
        valid_votes = [item for item in voted if item is not None]
        ensemble_result = {
            "sentences": valid_votes,
            "elapsed_seconds": sum(result.get("elapsed_seconds", 0.0) for result in model_results),
        }
        if len(valid_votes) != len(record["sentences"]):
            ensemble_result["parse_error"] = "No valid model vote for one or more sentence IDs"
        ensemble_results.append(ensemble_result)
    wall_elapsed = time.perf_counter() - run_started
    rows = evaluate_predictions("ensemble", records, ensemble_results)
    total_model_seconds = sum(model_elapsed.values())
    llm_calls = sum(len(record["sentences"]) for record in records) * len(detectors)
    metrics = metric_row("ensemble", rows, total_model_seconds, llm_calls)
    metrics["wall_clock_seconds"] = wall_elapsed
    for model_name, elapsed in model_elapsed.items():
        metrics[f"{model_name}_total_seconds"] = elapsed
        sentence_calls = sum(len(record["sentences"]) for record in records)
        metrics[f"{model_name}_average_seconds_per_paragraph"] = (
            elapsed / len(records) if records else None
        )
        metrics[f"{model_name}_average_seconds_per_sentence_call"] = (
            elapsed / sentence_calls if sentence_calls else None
        )
    print("\nTiming summary:")
    for model_name, elapsed in model_elapsed.items():
        print(
            f"  {model_name}: total={elapsed:.2f}s, "
            f"avg/paragraph={elapsed / len(records) if records else 0.0:.2f}s"
        )
    print(f"  ensemble wall clock: {wall_elapsed:.2f}s")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        pd.DataFrame([metrics]).to_excel(writer, index=False, sheet_name="Metrics")
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="PerSentence")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
