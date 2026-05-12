from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRIC_NAMES = ["MAE", "MAPE", "R2"]
BATCH_RE = re.compile(r"batch(?P<batch>\d+)-testbattery(?P<testbattery>\d+)$", re.IGNORECASE)
EXPERIMENT_RE = re.compile(r"experiment(?P<experiment>\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize result files under results/ with two-level averaging."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results"),
        help="Root directory that contains *_results.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_summary"),
        help="Directory for generated CSV and Markdown reports.",
    )
    return parser.parse_args()


def load_metrics(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        if "test_errors" in data:
            metrics = np.asarray(data["test_errors"], dtype=float).reshape(-1)
            if metrics.size >= 3:
                return metrics[:3]

        true_label = np.asarray(data["true_label"], dtype=float).reshape(-1)
        pred_label = np.asarray(data["pred_label"], dtype=float).reshape(-1)

    mae = np.mean(np.abs(true_label - pred_label))
    denom = np.where(true_label == 0, np.nan, true_label)
    mape = np.nanmean(np.abs((true_label - pred_label) / denom))
    ss_res = np.sum((true_label - pred_label) ** 2)
    ss_tot = np.sum((true_label - np.mean(true_label)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return np.array([mae, mape, r2], dtype=float)


def collect_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result_file in sorted(root.rglob("*_results.npz")):
        experiment_dir = result_file.parent
        battery_dir = experiment_dir.parent

        batch_match = BATCH_RE.fullmatch(battery_dir.name)
        exp_match = EXPERIMENT_RE.fullmatch(experiment_dir.name)
        if batch_match is None or exp_match is None:
            continue

        try:
            group = battery_dir.relative_to(root).parts[:-1]
        except ValueError:
            continue

        metrics = load_metrics(result_file)
        rows.append(
            {
                "group": "/".join(group) if group else root.name,
                "batch": int(batch_match.group("batch")),
                "testbattery": int(batch_match.group("testbattery")),
                "experiment": int(exp_match.group("experiment")),
                "file": str(result_file.relative_to(root)),
                **{name: float(value) for name, value in zip(METRIC_NAMES, metrics)},
            }
        )

    return rows


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"

    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, title: str, battery_df: pd.DataFrame, batch_df: pd.DataFrame) -> None:
    content = [
        f"# {title}",
        "",
        "Battery-level mean: mean of all experiments inside each `batchX-testbatteryY` folder.",
        "Batch-level mean: mean of the battery-level rows inside the same batch.",
        "",
        "## Battery Summary",
        "",
        dataframe_to_markdown(battery_df),
        "",
        "## Batch Summary",
        "",
        dataframe_to_markdown(batch_df),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(root)
    if not rows:
        raise SystemExit(f"No result files found under: {root}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["group", "batch", "testbattery", "experiment"]).reset_index(drop=True)

    battery_df = (
        df.groupby(["group", "batch", "testbattery"], as_index=False)
        .agg(
            n_experiments=("experiment", "count"),
            **{name: (name, "mean") for name in METRIC_NAMES},
        )
        .sort_values(["group", "batch", "testbattery"])
        .reset_index(drop=True)
    )

    batch_df = (
        battery_df.groupby(["group", "batch"], as_index=False)
        .agg(
            n_testbatteries=("testbattery", "count"),
            **{name: (name, "mean") for name in METRIC_NAMES},
        )
        .sort_values(["group", "batch"])
        .reset_index(drop=True)
    )

    battery_csv = output_dir / "battery_summary.csv"
    batch_csv = output_dir / "batch_summary.csv"
    report_md = output_dir / "summary_report.md"

    battery_df.to_csv(battery_csv, index=False, encoding="utf-8-sig")
    batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")
    write_report(report_md, "Results Summary", battery_df, batch_df)

    print(f"Scanned {len(df)} result files under {root}")
    print(f"Wrote battery summary: {battery_csv}")
    print(f"Wrote batch summary:   {batch_csv}")
    print(f"Wrote report:          {report_md}")
    print()
    print("Batch Summary")
    print(batch_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
