from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image


CHANNEL_NAMES = [
    "charge_time",
    "charge_current",
    "charge_voltage",
    "charge_temp",
    "discharge_time",
    "discharge_current",
    "discharge_voltage",
    "discharge_temp",
]

GROUP_NAMES = ["charge", "discharge"]
GROUP_SLICES = {
    "charge": slice(0, 4),
    "discharge": slice(4, 8),
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_map_frame() -> pd.DataFrame:
    rows = []
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        group = "charge" if channel_index < 4 else "discharge"
        rows.append(
            {
                "channel_index": channel_index,
                "channel_name": channel_name,
                "group": group,
                "position_in_group": channel_index % 4,
            }
        )
    return pd.DataFrame(rows)


def _take_evenly_spaced(sorted_indices: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.asarray([], dtype=int)
    sorted_indices = np.asarray(sorted_indices, dtype=int)
    if count >= len(sorted_indices):
        return sorted_indices.copy()

    positions = np.linspace(0, len(sorted_indices) - 1, num=count)
    positions = np.unique(np.round(positions).astype(int))
    if len(positions) < count:
        chosen = set(positions.tolist())
        extras = [idx for idx in range(len(sorted_indices)) if idx not in chosen]
        positions = np.sort(np.concatenate([positions, np.asarray(extras[: count - len(positions)], dtype=int)]))
    return sorted_indices[positions]


def stratified_select(
    indices: np.ndarray,
    scores: np.ndarray,
    counts: Sequence[int],
    descending: bool = True,
) -> np.ndarray:
    """
    Select samples by splitting sorted scores into len(counts) strata.

    The samples in each stratum are taken at evenly spaced positions to keep the
    selection deterministic and representative.
    """
    indices = np.asarray(indices, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if indices.shape[0] != scores.shape[0]:
        raise ValueError("indices and scores must have the same length")

    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    ordered_indices = indices[order]
    strata = np.array_split(ordered_indices, len(counts))

    selected = []
    for stratum_indices, count in zip(strata, counts):
        selected.extend(_take_evenly_spaced(stratum_indices, int(count)).tolist())

    return np.asarray(selected, dtype=int)


def pick_representative_indices(
    indices: np.ndarray,
    scores: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    low_mode: str = "low",
) -> np.ndarray:
    """
    Pick one high-score, one median-score, and one low-score sample.

    low_mode:
        - "low": choose the lowest-SOH sample
        - "error": choose the largest abs(error) sample if predictions are given
    """
    indices = np.asarray(indices, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if indices.size == 0:
        return indices

    high_idx = indices[np.argmax(scores[indices])]
    median_score = np.median(scores[indices])
    mid_idx = indices[np.argmin(np.abs(scores[indices] - median_score))]

    if low_mode == "error":
        if predictions is None:
            raise ValueError("predictions are required when low_mode='error'")
        low_idx = indices[np.argmax(np.abs(scores[indices] - predictions[indices]))]
    else:
        low_idx = indices[np.argmin(scores[indices])]

    ordered = []
    for idx in [high_idx, mid_idx, low_idx]:
        if idx not in ordered:
            ordered.append(int(idx))
    return np.asarray(ordered, dtype=int)


def raw_to_channel_signed(raw_values: np.ndarray) -> np.ndarray:
    raw_values = np.asarray(raw_values)
    if raw_values.ndim == 3:
        return raw_values.sum(axis=2)
    if raw_values.ndim == 2:
        return raw_values.sum(axis=1)
    raise ValueError(f"Unsupported raw_values shape: {raw_values.shape}")


def raw_to_channel_abs(raw_values: np.ndarray) -> np.ndarray:
    raw_values = np.asarray(raw_values)
    if raw_values.ndim == 3:
        return np.abs(raw_values).mean(axis=2)
    if raw_values.ndim == 2:
        return np.abs(raw_values).mean(axis=1)
    raise ValueError(f"Unsupported raw_values shape: {raw_values.shape}")


def channel_to_group(channel_values: np.ndarray) -> np.ndarray:
    channel_values = np.asarray(channel_values)
    if channel_values.ndim == 2:
        charge = channel_values[:, GROUP_SLICES["charge"]].mean(axis=1)
        discharge = channel_values[:, GROUP_SLICES["discharge"]].mean(axis=1)
        return np.stack([charge, discharge], axis=1)
    if channel_values.ndim == 1:
        charge = channel_values[GROUP_SLICES["charge"]].mean()
        discharge = channel_values[GROUP_SLICES["discharge"]].mean()
        return np.asarray([charge, discharge], dtype=float)
    raise ValueError(f"Unsupported channel_values shape: {channel_values.shape}")


def top_k_items(values: np.ndarray, names: Sequence[str], k: int = 3) -> List[Tuple[str, float]]:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)[::-1][:k]
    return [(str(names[idx]), float(values[idx])) for idx in order]


def stitch_vertical(
    image_paths: Sequence[Path],
    output_path: Path,
    padding: int = 24,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Path:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    canvas_width = max(widths)
    canvas_height = sum(heights) + padding * (len(images) - 1)

    canvas = Image.new("RGB", (canvas_width, canvas_height), background_color)
    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.height + padding

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    for img in images:
        img.close()
    return output_path
