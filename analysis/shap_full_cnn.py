from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analysis.shap_tools import (
    CHANNEL_NAMES,
    GROUP_NAMES,
    ensure_dir,
    feature_map_frame,
    pick_representative_indices,
    raw_to_channel_abs,
    raw_to_channel_signed,
    stratified_select,
    stitch_vertical,
    top_k_items,
)
from dataloader.XJTU_loader import XJTUDdataset
from nets.Model import SOHMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHAP explanation for Unified-Full-CNN")
    parser.add_argument("--data", type=str, default="XJTU", choices=["XJTU"])
    parser.add_argument("--input_type", type=str, default="full", choices=["full"])
    parser.add_argument("--model", type=str, default="CNN", choices=["CNN"])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--test_battery_id", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--random_seed", type=int, default=2023)
    parser.add_argument("--normalized_type", type=str, default="minmax", choices=["minmax", "standard"])
    parser.add_argument("--minmax_range", nargs=2, type=float, default=[-1.0, 1.0])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--feature_channels", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=30)
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--output_root", type=Path, default=Path("results/shap"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint_selection",
        type=str,
        default="best_valid",
        choices=["explicit", "best_valid", "median_valid"],
    )
    parser.add_argument("--experiment", type=int, default=1)
    parser.add_argument("--background_size", type=int, default=32)
    parser.add_argument("--explain_size", type=int, default=64)
    parser.add_argument("--background_strata", nargs=3, type=int, default=[8, 16, 8])
    parser.add_argument("--explain_strata", nargs=3, type=int, default=[16, 32, 16])
    parser.add_argument("--local_mode", type=str, default="low", choices=["low", "error"])
    parser.add_argument("--explainer", type=str, default="auto", choices=["auto", "deep", "gradient"])
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        return "cpu"
    return device


def build_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    result_dir = (
        args.results_root
        / f"{args.data}-{args.input_type}"
        / args.model
        / f"batch{args.batch}-testbattery{args.test_battery_id}"
    )
    if args.output_dir is None:
        output_dir = (
            args.output_root
            / f"{args.data}-{args.input_type}"
            / args.model
            / f"batch{args.batch}-testbattery{args.test_battery_id}"
        )
    else:
        output_dir = args.output_dir
    return result_dir, output_dir


def load_data_bundle(args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    loader = XJTUDdataset(args)
    return loader.get_full_arrays(test_battery_id=args.test_battery_id)


def load_model(args: argparse.Namespace, checkpoint_path: Path) -> SOHMode:
    model = SOHMode(args)
    model.load_checkpoint(checkpoint_path, map_location=args.device)
    return model


def predict_numpy(model: torch.nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    with torch.no_grad():
        pred = model(tensor)
    return pred.detach().cpu().numpy().reshape(-1)


def normalize_shap_values(shap_values: object, expected_shape: Tuple[int, ...]) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) == 0:
            raise ValueError("Empty SHAP value list returned by the explainer.")
        shap_values = shap_values[0]

    array = np.asarray(shap_values)
    if array.ndim == len(expected_shape) + 1 and array.shape[0] == 1:
        array = array[0]
    if array.shape != expected_shape:
        raise ValueError(f"Unexpected SHAP shape: {array.shape}, expected {expected_shape}")
    return array


def extract_scalar(value: object) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("Failed to extract scalar from empty value.")
    return float(array[0])


def compute_shap_values(
    model: torch.nn.Module,
    background_x: np.ndarray,
    explain_x: np.ndarray,
    device: str,
    explainer_mode: str,
) -> Tuple[np.ndarray, float, str]:
    try:
        import shap  # local import: optional dependency
    except ImportError as exc:
        raise ImportError(
            "The `shap` package is required for this analysis. Install it with `pip install shap`."
        ) from exc

    background_tensor = torch.from_numpy(np.asarray(background_x, dtype=np.float32)).to(device)
    explain_tensor = torch.from_numpy(np.asarray(explain_x, dtype=np.float32)).to(device)
    expected_shape = tuple(explain_x.shape)

    def _try_deep() -> Tuple[np.ndarray, float]:
        explainer = shap.DeepExplainer(model, background_tensor)
        try:
            shap_values = explainer.shap_values(explain_tensor, check_additivity=False)
        except TypeError:
            shap_values = explainer.shap_values(explain_tensor)
        expected_value = extract_scalar(explainer.expected_value)
        return normalize_shap_values(shap_values, expected_shape), expected_value

    def _try_gradient() -> Tuple[np.ndarray, float]:
        explainer = shap.GradientExplainer(model, background_tensor)
        shap_values = explainer.shap_values(explain_tensor)
        expected_value = getattr(explainer, "expected_value", None)
        if expected_value is None:
            with torch.no_grad():
                expected_value = model(background_tensor).detach().cpu().numpy().mean()
        return normalize_shap_values(shap_values, expected_shape), extract_scalar(expected_value)

    if explainer_mode in ["auto", "deep"]:
        try:
            shap_values, expected_value = _try_deep()
            return shap_values, expected_value, "DeepExplainer"
        except Exception as exc:
            if explainer_mode == "deep":
                raise
            print(f"DeepExplainer failed, falling back to GradientExplainer: {exc}")

    shap_values, expected_value = _try_gradient()
    return shap_values, expected_value, "GradientExplainer"


def make_explanation(values: np.ndarray, data: np.ndarray, feature_names: Sequence[str], base_value: float):
    import shap  # local import: optional dependency

    values = np.asarray(values, dtype=float)
    data = np.asarray(data, dtype=float)
    if values.ndim == 2:
        base_values = np.full(values.shape[0], base_value, dtype=float)
    else:
        base_values = float(base_value)
    return shap.Explanation(
        values=values,
        data=data,
        feature_names=list(feature_names),
        base_values=base_values,
    )


def save_current_figure(path: Path, dpi: int = 300) -> None:
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_group_bar(group_exp, output_path: Path) -> None:
    import shap

    plt.figure(figsize=(6.5, 4.2))
    shap.plots.bar(group_exp, max_display=2, show=False)
    plt.title("Grouped SHAP Importance: Charge vs Discharge")
    save_current_figure(output_path)


def plot_channel_bar(channel_exp, output_path: Path) -> None:
    import shap

    plt.figure(figsize=(9.5, 5.5))
    shap.plots.bar(channel_exp, max_display=8, show=False)
    plt.title("Channel-level SHAP Importance")
    save_current_figure(output_path)


def plot_beeswarm(channel_exp, output_path: Path) -> None:
    import shap

    plt.figure(figsize=(10.5, 6.2))
    shap.plots.beeswarm(channel_exp, max_display=8, show=False)
    plt.title("Channel-level SHAP Beeswarm")
    save_current_figure(output_path)


def plot_heatmap(channel_exp, output_path: Path) -> None:
    import shap

    plt.figure(figsize=(11.5, 7.2))
    shap.plots.heatmap(channel_exp, show=False)
    plt.title("Channel-level SHAP Heatmap")
    save_current_figure(output_path)


def plot_waterfall_triptych(
    channel_exp,
    output_path: Path,
    sample_indices: Sequence[int],
    sample_titles: Sequence[str],
) -> None:
    import shap

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        temp_paths: List[Path] = []
        for idx, title in zip(sample_indices, sample_titles):
            plt.figure(figsize=(10.5, 5.8))
            shap.plots.waterfall(channel_exp[int(idx)], max_display=8, show=False)
            plt.title(title)
            temp_path = tmpdir_path / f"waterfall_{len(temp_paths)}.png"
            save_current_figure(temp_path)
            temp_paths.append(temp_path)
        stitch_vertical(temp_paths, output_path)


def build_selection_frame(
    meta: pd.DataFrame,
    indices: np.ndarray,
    scores: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    selection_name: str = "selection",
    selection_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    frame = meta.iloc[indices].reset_index(drop=True).copy()
    frame.insert(0, "sample_index", np.asarray(indices, dtype=int))
    frame.insert(1, "selection", selection_name)
    frame["soh"] = np.asarray(scores, dtype=float)[indices]
    if predictions is not None:
        preds = np.asarray(predictions, dtype=float)
        if preds.shape[0] == len(scores):
            frame["prediction"] = preds[indices]
        elif preds.shape[0] == len(indices):
            frame["prediction"] = preds
        else:
            raise ValueError(
                "predictions must either match the full score array length or the selected index length"
            )
        frame["abs_error"] = np.abs(frame["prediction"] - frame["soh"])
    if selection_labels is not None:
        labels = list(selection_labels)
        if len(labels) != len(frame):
            raise ValueError("selection_labels must match the number of selected samples")
        frame["selection_stratum"] = labels
    return frame


def build_grouped_shap_frame(
    meta: pd.DataFrame,
    indices: np.ndarray,
    true_soh: np.ndarray,
    pred_soh: np.ndarray,
    channel_signed: np.ndarray,
    channel_abs: np.ndarray,
) -> pd.DataFrame:
    group_signed = np.column_stack([channel_signed[:, :4].sum(axis=1), channel_signed[:, 4:].sum(axis=1)])
    group_abs = np.column_stack([channel_abs[:, :4].mean(axis=1), channel_abs[:, 4:].mean(axis=1)])

    frame = meta.iloc[indices].reset_index(drop=True).copy()
    frame.insert(0, "sample_index", np.asarray(indices, dtype=int))
    true_array = np.asarray(true_soh, dtype=float)
    pred_array = np.asarray(pred_soh, dtype=float)
    if true_array.shape[0] == len(indices):
        frame["true_soh"] = true_array
    else:
        frame["true_soh"] = true_array[indices]
    if pred_array.shape[0] == len(indices):
        frame["pred_soh"] = pred_array
    else:
        frame["pred_soh"] = pred_array[indices]
    frame["abs_error"] = np.abs(frame["pred_soh"] - frame["true_soh"])
    frame["charge_group_abs"] = group_abs[:, 0]
    frame["discharge_group_abs"] = group_abs[:, 1]
    frame["charge_group_signed"] = group_signed[:, 0]
    frame["discharge_group_signed"] = group_signed[:, 1]

    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        frame[f"{channel_name}_abs"] = channel_abs[:, channel_index]
        frame[f"{channel_name}_signed"] = channel_signed[:, channel_index]

    return frame


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    checkpoint_info: Dict[str, object],
    explanation_backend: str,
    background_indices: np.ndarray,
    explain_indices: np.ndarray,
    local_indices: np.ndarray,
    local_roles: Sequence[str],
    group_abs_mean: np.ndarray,
    channel_abs_mean: np.ndarray,
) -> None:
    top_channels = top_k_items(channel_abs_mean, CHANNEL_NAMES, k=3)
    group_pairs = [(GROUP_NAMES[i], float(group_abs_mean[i])) for i in np.argsort(group_abs_mean)[::-1]]

    lines = [
        "# Unified-Full-CNN SHAP Summary",
        "",
        f"- Data: `{args.data}`",
        f"- Input type: `{args.input_type}`",
        f"- Batch: `{args.batch}`",
        f"- Test battery: `{args.test_battery_id}`",
        f"- Checkpoint selection: `{checkpoint_info.get('selection_strategy', 'explicit')}`",
        f"- Checkpoint: `{checkpoint_info.get('checkpoint_path')}`",
        f"- Result file: `{checkpoint_info.get('result_path')}`",
        f"- SHAP backend: `{explanation_backend}`",
        f"- Background samples: `{len(background_indices)}`",
        f"- Explain samples: `{len(explain_indices)}`",
        f"- Local samples: `{len(local_indices)}` ({', '.join(local_roles)})",
        "",
        "## Global Importance",
        "",
        "| Group | Mean Abs SHAP |",
        "| --- | ---: |",
    ]
    for name, value in group_pairs:
        lines.append(f"| {name} | {value:.6f} |")
    lines.extend(
        [
            "",
            "| Channel | Mean Abs SHAP |",
            "| --- | ---: |",
        ]
    )
    for name, value in top_channels:
        lines.append(f"| {name} | {value:.6f} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Background samples are drawn from the training split with a fixed 8/16/8 stratification.",
            "- Explain samples are drawn from the held-out test battery with a fixed 16/32/16 stratification.",
            "- Channel-level signed SHAP values are time-summed; importance scores use absolute-value means.",
            "- The waterfall figure uses the three representative samples selected from the explained test set.",
            "",
        ]
    )

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def resolve_checkpoint(args: argparse.Namespace, result_dir: Path) -> Dict[str, object]:
    prefix = f"{args.model}_{args.input_type}"

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.resolve()
        result_path = checkpoint_path.with_name(f"{prefix}_results.npz")
        info: Dict[str, object] = {
            "selection_strategy": "explicit_path",
            "checkpoint_path": str(checkpoint_path),
            "result_path": str(result_path) if result_path.exists() else None,
        }
        if result_path.exists():
            with np.load(result_path, allow_pickle=False) as report:
                info["best_valid_loss"] = float(np.min(np.asarray(report["valid_loss"], dtype=float)))
                if "test_errors" in report:
                    errors = np.asarray(report["test_errors"], dtype=float).reshape(-1)
                    if errors.size >= 3:
                        info["test_mae"] = float(errors[0])
                        info["test_mape"] = float(errors[1])
                        info["test_r2"] = float(errors[2])
        return info

    if args.checkpoint_selection == "explicit":
        exp_dir = result_dir / f"experiment{args.experiment}"
        checkpoint_path = exp_dir / f"{prefix}_model.pkl"
        result_path = exp_dir / f"{prefix}_results.npz"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        info = {
            "selection_strategy": f"experiment{args.experiment}",
            "checkpoint_path": str(checkpoint_path),
            "result_path": str(result_path) if result_path.exists() else None,
        }
        if result_path.exists():
            with np.load(result_path, allow_pickle=False) as report:
                info["best_valid_loss"] = float(np.min(np.asarray(report["valid_loss"], dtype=float)))
                if "test_errors" in report:
                    errors = np.asarray(report["test_errors"], dtype=float).reshape(-1)
                    if errors.size >= 3:
                        info["test_mae"] = float(errors[0])
                        info["test_mape"] = float(errors[1])
                        info["test_r2"] = float(errors[2])
        return info

    candidates = []
    for exp_dir in sorted(result_dir.glob("experiment*")):
        checkpoint_path = exp_dir / f"{prefix}_model.pkl"
        result_path = exp_dir / f"{prefix}_results.npz"
        if not checkpoint_path.exists() or not result_path.exists():
            continue
        with np.load(result_path, allow_pickle=False) as report:
            valid_loss = np.asarray(report["valid_loss"], dtype=float)
            best_valid_loss = float(np.min(valid_loss))
            candidate = {
                "experiment_dir": str(exp_dir),
                "checkpoint_path": str(checkpoint_path),
                "result_path": str(result_path),
                "best_valid_loss": best_valid_loss,
            }
            if "test_errors" in report:
                errors = np.asarray(report["test_errors"], dtype=float).reshape(-1)
                if errors.size >= 3:
                    candidate["test_mae"] = float(errors[0])
                    candidate["test_mape"] = float(errors[1])
                    candidate["test_r2"] = float(errors[2])
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(f"No experiment checkpoints found under: {result_dir}")

    if args.checkpoint_selection == "best_valid":
        chosen = min(candidates, key=lambda item: item["best_valid_loss"])
        strategy = "best_valid"
    else:
        median_score = float(np.median([item["best_valid_loss"] for item in candidates]))
        chosen = min(candidates, key=lambda item: abs(item["best_valid_loss"] - median_score))
        strategy = "median_valid"

    chosen["selection_strategy"] = strategy
    return chosen


def validate_strata(background_strata: Sequence[int], explain_strata: Sequence[int], background_size: int, explain_size: int) -> None:
    if sum(background_strata) != background_size:
        raise ValueError(f"background strata must sum to {background_size}, got {sum(background_strata)}")
    if sum(explain_strata) != explain_size:
        raise ValueError(f"explain strata must sum to {explain_size}, got {sum(explain_strata)}")


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    args.minmax_range = tuple(args.minmax_range)
    args.background_strata = tuple(int(x) for x in args.background_strata)
    args.explain_strata = tuple(int(x) for x in args.explain_strata)

    validate_strata(args.background_strata, args.explain_strata, args.background_size, args.explain_size)
    set_random_seed(args.random_seed)

    result_dir, output_dir = build_paths(args)
    ensure_dir(output_dir)

    checkpoint_info = resolve_checkpoint(args, result_dir)
    checkpoint_path = Path(checkpoint_info["checkpoint_path"])

    data_bundle = load_data_bundle(args)
    train_split = data_bundle["train"]
    test_split = data_bundle["test"]

    train_x = np.asarray(train_split["x"], dtype=np.float32)
    train_y = np.asarray(train_split["y"], dtype=np.float32).reshape(-1)
    train_meta = train_split["meta"].reset_index(drop=True)
    test_x = np.asarray(test_split["x"], dtype=np.float32)
    test_y = np.asarray(test_split["y"], dtype=np.float32).reshape(-1)
    test_meta = test_split["meta"].reset_index(drop=True)

    background_indices = stratified_select(np.arange(len(train_y)), train_y, args.background_strata, descending=True)
    explain_indices = stratified_select(np.arange(len(test_y)), test_y, args.explain_strata, descending=True)

    background_labels = ["high"] * args.background_strata[0] + ["mid"] * args.background_strata[1] + ["low"] * args.background_strata[2]
    explain_labels = ["high"] * args.explain_strata[0] + ["mid"] * args.explain_strata[1] + ["low"] * args.explain_strata[2]
    if len(background_indices) != len(background_labels) or len(explain_indices) != len(explain_labels):
        raise RuntimeError("Stratified selection produced an unexpected number of samples.")

    model = load_model(args, checkpoint_path)

    test_pred_all = predict_numpy(model, test_x, args.device)
    explain_pred = test_pred_all[explain_indices]
    explain_true = test_y[explain_indices]

    if args.local_mode == "error":
        local_indices = pick_representative_indices(explain_indices, test_y, predictions=test_pred_all, low_mode="error")
        local_roles = ["high", "mid", "error_case"]
    else:
        local_indices = pick_representative_indices(explain_indices, test_y, low_mode="low")
        local_roles = ["high", "mid", "low"]
    if len(local_indices) < 3:
        # Keep the output deterministic even if the set collapses.
        padded = list(local_indices.tolist())
        while len(padded) < 3 and len(padded) > 0:
            padded.append(padded[-1])
        local_indices = np.asarray(padded, dtype=int)

    background_x = train_x[background_indices]
    explain_x = test_x[explain_indices]

    raw_shap_values, expected_value, explainer_name = compute_shap_values(
        model=model,
        background_x=background_x,
        explain_x=explain_x,
        device=args.device,
        explainer_mode=args.explainer,
    )

    channel_signed = raw_to_channel_signed(raw_shap_values)
    channel_abs = raw_to_channel_abs(raw_shap_values)
    group_signed = np.column_stack([channel_signed[:, :4].sum(axis=1), channel_signed[:, 4:].sum(axis=1)])
    group_abs = np.column_stack([channel_abs[:, :4].mean(axis=1), channel_abs[:, 4:].mean(axis=1)])
    channel_input = explain_x.mean(axis=2)
    group_input = np.column_stack([explain_x[:, :4, :].mean(axis=(1, 2)), explain_x[:, 4:, :].mean(axis=(1, 2))])

    import shap

    group_exp = shap.Explanation(
        values=group_abs,
        data=group_input,
        feature_names=GROUP_NAMES,
        base_values=np.zeros(group_abs.shape[0], dtype=float),
    )
    channel_abs_exp = shap.Explanation(
        values=channel_abs,
        data=channel_input,
        feature_names=CHANNEL_NAMES,
        base_values=np.zeros(channel_abs.shape[0], dtype=float),
    )
    channel_signed_exp = shap.Explanation(
        values=channel_signed,
        data=channel_input,
        feature_names=CHANNEL_NAMES,
        base_values=np.full(channel_signed.shape[0], expected_value, dtype=float),
    )

    # Save tabular artifacts first.
    feature_map = feature_map_frame()
    feature_map.to_csv(output_dir / "feature_map.csv", index=False, encoding="utf-8-sig")

    background_frame = build_selection_frame(
        meta=train_meta,
        indices=background_indices,
        scores=train_y,
        selection_name="background",
        selection_labels=background_labels,
    )
    background_frame.to_csv(output_dir / "background_idx.csv", index=False, encoding="utf-8-sig")

    explain_frame = build_selection_frame(
        meta=test_meta,
        indices=explain_indices,
        scores=test_y,
        predictions=np.asarray(test_pred_all, dtype=float),
        selection_name="explain",
        selection_labels=explain_labels,
    )
    explain_frame["is_local_sample"] = False
    explain_frame["local_role"] = ""
    for local_index, local_role in zip(local_indices.tolist(), local_roles):
        explain_frame.loc[explain_frame["sample_index"] == local_index, "is_local_sample"] = True
        explain_frame.loc[explain_frame["sample_index"] == local_index, "local_role"] = local_role
    explain_frame.to_csv(output_dir / "explain_idx.csv", index=False, encoding="utf-8-sig")

    grouped_frame = build_grouped_shap_frame(
        meta=test_meta,
        indices=explain_indices,
        true_soh=test_y,
        pred_soh=test_pred_all,
        channel_signed=channel_signed,
        channel_abs=channel_abs,
    )
    grouped_frame.to_csv(output_dir / "shap_values_grouped.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        output_dir / "shap_values_raw.npz",
        raw_shap_values=raw_shap_values,
        channel_signed=channel_signed,
        channel_abs=channel_abs,
        group_signed=group_signed,
        group_abs=group_abs,
        background_indices=background_indices,
        explain_indices=explain_indices,
        local_indices=local_indices,
        background_x=background_x,
        explain_x=explain_x,
        explain_true=explain_true,
        explain_pred=np.asarray(explain_pred, dtype=float),
        expected_value=np.asarray([expected_value], dtype=float),
        explainer_name=np.asarray([explainer_name]),
        checkpoint_path=np.asarray([str(checkpoint_path)]),
    )

    # Save figures.
    plot_group_bar(group_exp, output_dir / "fig01_group_bar.png")
    plot_channel_bar(channel_abs_exp, output_dir / "fig02_channel_bar.png")
    plot_beeswarm(channel_signed_exp, output_dir / "fig03_beeswarm.png")
    plot_heatmap(channel_signed_exp, output_dir / "fig04_heatmap.png")

    local_titles = []
    for local_index, local_role in zip(local_indices.tolist(), local_roles):
        row = test_meta.iloc[int(local_index)]
        local_titles.append(
            f"{local_role.title()} | battery {int(row['battery_id'])} | cycle {int(row['cycle_id'])} | SOH {float(row['soh']):.4f}"
        )
    plot_waterfall_triptych(
        channel_signed_exp,
        output_dir / "fig05_waterfall_triptych.png",
        local_indices,
        local_titles,
    )

    group_abs_mean = group_abs.mean(axis=0)
    channel_abs_mean = channel_abs.mean(axis=0)
    write_summary(
        output_dir=output_dir,
        args=args,
        checkpoint_info=checkpoint_info,
        explanation_backend=explainer_name,
        background_indices=background_indices,
        explain_indices=explain_indices,
        local_indices=local_indices,
        local_roles=local_roles,
        group_abs_mean=group_abs_mean,
        channel_abs_mean=channel_abs_mean,
    )

    print(f"SHAP explanation saved to: {output_dir}")
    print(f"Checkpoint used: {checkpoint_path}")
    print(f"Explainer backend: {explainer_name}")


if __name__ == "__main__":
    main()
