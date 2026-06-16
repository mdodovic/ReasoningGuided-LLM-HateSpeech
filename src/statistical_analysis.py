"""Statistical summaries for paired paragraph-context predictions."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score


CORRECTNESS_COLUMNS = {
    "binary": "binary_correct",
    "category": "category_correct",
    "full_code": "full_code_correct",
}


def summary_statistics(sentence_rows: list[dict]) -> list[dict]:
    """Calculate post-hoc performance and runtime summaries by model."""
    df = pd.DataFrame(sentence_rows)
    if df.empty:
        return []
    output = []
    for model, model_df in df.groupby("model", sort=False):
        valid = model_df[model_df["prediction_valid"] == True].copy()  # noqa: E712
        paragraph_times = model_df.groupby("paragraph_id")["paragraph_seconds"].first().dropna().astype(float)
        row = {
            "model": model,
            "sentences_total": len(model_df),
            "sentences_valid": len(valid),
            "sentences_invalid": len(model_df) - len(valid),
            "binary_accuracy": valid["binary_correct"].astype(float).mean() if not valid.empty else None,
            "category_accuracy": valid["category_correct"].astype(float).mean() if not valid.empty else None,
            "full_code_accuracy": valid["full_code_correct"].astype(float).mean() if not valid.empty else None,
            "total_inference_seconds": paragraph_times.sum() if not paragraph_times.empty else None,
            "mean_seconds_per_paragraph": paragraph_times.mean() if not paragraph_times.empty else None,
            "median_seconds_per_paragraph": paragraph_times.median() if not paragraph_times.empty else None,
            "standard_deviation_seconds_per_paragraph": paragraph_times.std(ddof=1) if len(paragraph_times) > 1 else 0.0,
        }
        if not valid.empty:
            binary_true = valid["gt_has_hate_speech"].astype(int)
            binary_pred = valid["pred_has_hate_speech"].astype(int)
            category_true = valid["gt_category"].astype(int)
            category_pred = valid["pred_category"].astype(int)
            row.update({
                "binary_precision": precision_score(binary_true, binary_pred, zero_division=0),
                "binary_recall": recall_score(binary_true, binary_pred, zero_division=0),
                "binary_f1": f1_score(binary_true, binary_pred, zero_division=0),
                "binary_balanced_accuracy": (
                    balanced_accuracy_score(binary_true, binary_pred)
                    if binary_true.nunique() > 1
                    else (binary_true == binary_pred).mean()
                ),
                "category_macro_f1": f1_score(category_true, category_pred, average="macro", zero_division=0),
            })
        output.append(row)
    return output


def model_statistics(
    sentence_rows: list[dict], bootstrap_iterations: int = 2000, seed: int = 42
) -> list[dict]:
    """Return paragraph-level SD and cluster-bootstrap confidence intervals."""
    df = pd.DataFrame(sentence_rows)
    if df.empty:
        return []
    rng = np.random.default_rng(seed)
    output = []
    for model, model_df in df.groupby("model", sort=False):
        valid = model_df[model_df["prediction_valid"] == True].copy()  # noqa: E712
        for task, column in CORRECTNESS_COLUMNS.items():
            task_df = valid[valid[column].notna()]
            if task_df.empty:
                continue
            cluster = task_df.groupby("paragraph_id")[column].agg(["sum", "count"]).astype(float)
            paragraph_scores = cluster["sum"] / cluster["count"]
            cluster_sums = cluster["sum"].to_numpy()
            cluster_counts = cluster["count"].to_numpy()
            boot = np.empty(bootstrap_iterations, dtype=float)
            for index in range(bootstrap_iterations):
                sampled = rng.integers(0, len(cluster), size=len(cluster))
                boot[index] = cluster_sums[sampled].sum() / cluster_counts[sampled].sum()
            ci_low = np.quantile(boot, 0.025)
            ci_high = np.quantile(boot, 0.975)
            output.append({
                "model": model,
                "task": task,
                "sentence_accuracy": task_df[column].astype(float).mean(),
                "paragraph_mean_accuracy": paragraph_scores.mean(),
                "paragraph_standard_deviation": paragraph_scores.std(ddof=1) if len(paragraph_scores) > 1 else 0.0,
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "bootstrap_95_ci_half_width": (ci_high - ci_low) / 2,
                "bootstrap_iterations": bootstrap_iterations,
                "valid_sentences": len(task_df),
                "valid_paragraphs": len(paragraph_scores),
            })
    return output


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, original_index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[original_index])
        running = max(running, value)
        adjusted[original_index] = running
    return adjusted


def pairwise_mcnemar(sentence_rows: list[dict]) -> list[dict]:
    """Run exact paired McNemar tests on common valid sentence predictions."""
    df = pd.DataFrame(sentence_rows)
    if df.empty or df["model"].nunique() < 2:
        return []
    results = []
    models = list(df["model"].drop_duplicates())
    key_columns = ["paragraph_id", "sentence_id"]
    for model_a, model_b in combinations(models, 2):
        left = df[(df["model"] == model_a) & (df["prediction_valid"] == True)]  # noqa: E712
        right = df[(df["model"] == model_b) & (df["prediction_valid"] == True)]  # noqa: E712
        for task, column in CORRECTNESS_COLUMNS.items():
            left_task = left
            right_task = right
            paired = left_task[key_columns + [column]].merge(
                right_task[key_columns + [column]], on=key_columns, suffixes=("_a", "_b")
            ).dropna()
            if paired.empty:
                continue
            a_correct = paired[f"{column}_a"].astype(bool)
            b_correct = paired[f"{column}_b"].astype(bool)
            a_only = int((a_correct & ~b_correct).sum())
            b_only = int((~a_correct & b_correct).sum())
            discordant = a_only + b_only
            p_value = binomtest(min(a_only, b_only), discordant, 0.5).pvalue if discordant else 1.0
            results.append({
                "model_a": model_a,
                "model_b": model_b,
                "task": task,
                "common_valid_sentences": len(paired),
                "model_a_accuracy": a_correct.mean(),
                "model_b_accuracy": b_correct.mean(),
                "accuracy_difference_a_minus_b": a_correct.mean() - b_correct.mean(),
                "a_correct_b_wrong": a_only,
                "a_wrong_b_correct": b_only,
                "discordant_pairs": discordant,
                "exact_p_value": p_value,
            })
    adjusted = _holm_adjust([row["exact_p_value"] for row in results])
    for row, adjusted_p in zip(results, adjusted):
        row["holm_adjusted_p_value"] = adjusted_p
        row["significant_at_0_05"] = adjusted_p < 0.05
    return results
