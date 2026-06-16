"""Calculate statistics from completed paragraph-context inference results."""

from pathlib import Path

import pandas as pd

from src.statistical_analysis import model_statistics, pairwise_mcnemar, summary_statistics

REPO_ROOT = Path(__file__).resolve().parent
INPUT_PATH = REPO_ROOT / "results" / "paragraph_context_comparison.xlsx"
OUTPUT_PATH = REPO_ROOT / "results" / "paragraph_context_statistics.xlsx"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Inference results not found: {INPUT_PATH}. Run paragraph_context_run.py first."
        )
    rows = pd.read_excel(INPUT_PATH, sheet_name="PerSentence").to_dict("records")
    summary_rows = summary_statistics(rows)
    statistics_rows = model_statistics(rows)
    mcnemar_rows = pairwise_mcnemar(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="Summary")
        pd.DataFrame(statistics_rows).to_excel(writer, index=False, sheet_name="Statistics")
        pd.DataFrame(mcnemar_rows).to_excel(writer, index=False, sheet_name="McNemar")

    print(f"Statistics saved to: {OUTPUT_PATH}")
    for row in mcnemar_rows:
        print(
            f"{row['model_a']} vs {row['model_b']} [{row['task']}]: "
            f"Holm p={row['holm_adjusted_p_value']:.4g}, "
            f"significant={row['significant_at_0_05']}"
        )


if __name__ == "__main__":
    main()
