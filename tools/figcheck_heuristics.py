"""FigCheck-inspired heuristic scoring utilities.

These helpers provide optional scoring signals that approximate the techniques described in the
public FigCheck repository. The implementation is self-contained so that we can iterate on the
ideas without shipping upstream code. Callers should guard use with feature flags because the
heuristics are intended for experimentation during A/B comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:  # tqdm is optional; fall back to a no-op iterator when unavailable
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is missing
    tqdm = None

__all__ = [
    "FigcheckHeuristicConfig",
    "apply_figcheck_scores",
    "score_duplicate_figcheck_style",
]


@dataclass
class FigcheckHeuristicConfig:
    """Tunable parameters for the FigCheck-inspired heuristics."""

    enable_band_alignment: bool = True
    enable_contrast_normalization: bool = True
    enable_partial_template: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    lane_kernel: int = 9
    adaptive_block_size: int = 31
    adaptive_c: int = -5
    rotation_degrees: Tuple[int, ...] = (0, 90)
    min_partial_scale: float = 0.45
    partial_template_min_size: int = 64
    profile_eps: float = 1e-6
    score_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    dtype: np.dtype = np.float32

    def __post_init__(self) -> None:
        self.adaptive_block_size = _ensure_odd(self.adaptive_block_size)
        self.lane_kernel = max(3, self.lane_kernel)


def score_duplicate_figcheck_style(
    path_a: Path | str,
    path_b: Path | str,
    config: Optional[FigcheckHeuristicConfig] = None,
) -> dict:
    """Compute FigCheck-inspired duplicate signals for a pair of panels."""

    cfg = config or FigcheckHeuristicConfig()
    path_a = Path(path_a)
    path_b = Path(path_b)

    img_a = cv2.imread(str(path_a), cv2.IMREAD_COLOR)
    img_b = cv2.imread(str(path_b), cv2.IMREAD_COLOR)

    if img_a is None or img_b is None:
        missing = []
        if img_a is None:
            missing.append(str(path_a))
        if img_b is None:
            missing.append(str(path_b))
        return {
            "score": np.nan,
            "band_alignment": np.nan,
            "projection_corr": np.nan,
            "lane_iou": np.nan,
            "partial_score": np.nan,
            "rotation": None,
            "error": f"missing_image:{','.join(missing)}",
        }

    base_a = _preprocess_gray(img_a, cfg)
    base_b = _preprocess_gray(img_b, cfg)

    evaluations: List[dict] = []
    for deg in cfg.rotation_degrees:
        rot_a = _rotate_gray(base_a, deg)
        rot_b = _rotate_gray(base_b, deg)
        axis = 0 if deg % 180 == 0 else 1

        band_score = 0.0
        lane_iou = 0.0
        if cfg.enable_band_alignment:
            mask_a = _lane_mask(rot_a, axis, cfg)
            mask_b = _lane_mask(rot_b, axis, cfg)
            profile_a = _lane_profile(mask_a, axis)
            profile_b = _lane_profile(mask_b, axis)
            band_score = _projection_alignment(profile_a, profile_b, cfg)
            lane_iou = _lane_iou(mask_a, mask_b)

        proj_a = rot_a.mean(axis=axis)
        proj_b = rot_b.mean(axis=axis)
        projection_corr = _projection_alignment(proj_a, proj_b, cfg)

        partial_score = 0.0
        if cfg.enable_partial_template:
            partial_score = _partial_template_score(rot_a, rot_b, cfg)

        composite = (
            cfg.score_weights[0] * band_score
            + cfg.score_weights[1] * projection_corr
            + cfg.score_weights[2] * partial_score
        )

        evaluations.append(
            {
                "rotation": deg,
                "band_alignment": float(band_score),
                "projection_corr": float(projection_corr),
                "lane_iou": float(lane_iou),
                "partial_score": float(partial_score),
                "composite": float(composite),
            }
        )

    if not evaluations:
        return {
            "score": np.nan,
            "band_alignment": np.nan,
            "projection_corr": np.nan,
            "lane_iou": np.nan,
            "partial_score": np.nan,
            "rotation": None,
            "error": "no_evaluations",
        }

    best = max(evaluations, key=lambda item: item["composite"])
    result = {
        "score": best["composite"],
        "band_alignment": best["band_alignment"],
        "projection_corr": best["projection_corr"],
        "lane_iou": best["lane_iou"],
        "partial_score": best["partial_score"],
        "rotation": best["rotation"],
        "evaluations": evaluations,
    }
    return result


def apply_figcheck_scores(
    df: pd.DataFrame,
    config: Optional[FigcheckHeuristicConfig] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Annotate a duplicate DataFrame with FigCheck-inspired scores."""

    if df is None or df.empty:
        return df

    cfg = config or FigcheckHeuristicConfig()
    df = df.copy()

    columns = [
        "FigCheck_BandScore",
        "FigCheck_ProjectionCorr",
        "FigCheck_LaneIoU",
        "FigCheck_PartialScore",
        "FigCheck_NormalizedScore",
        "FigCheck_Rotation",
        "FigCheck_Error",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan

    iterator = df.iterrows()
    if progress and tqdm is not None:
        iterator = tqdm(df.iterrows(), total=len(df), desc="FigCheck Heuristics")

    for idx, row in iterator:
        path_a = row.get("Path_A")
        path_b = row.get("Path_B")
        if not path_a or not path_b:
            df.at[idx, "FigCheck_Error"] = "missing_paths"
            continue

        try:
            scores = score_duplicate_figcheck_style(path_a, path_b, cfg)
        except Exception as exc:  # pragma: no cover - defensive fallback
            df.at[idx, "FigCheck_Error"] = f"exception:{exc}"
            continue

        df.at[idx, "FigCheck_BandScore"] = scores.get("band_alignment")
        df.at[idx, "FigCheck_ProjectionCorr"] = scores.get("projection_corr")
        df.at[idx, "FigCheck_LaneIoU"] = scores.get("lane_iou")
        df.at[idx, "FigCheck_PartialScore"] = scores.get("partial_score")
        df.at[idx, "FigCheck_NormalizedScore"] = scores.get("score")
        df.at[idx, "FigCheck_Rotation"] = scores.get("rotation")
        if scores.get("error"):
            df.at[idx, "FigCheck_Error"] = scores["error"]

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _preprocess_gray(img_bgr: np.ndarray, cfg: FigcheckHeuristicConfig) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if cfg.enable_contrast_normalization:
        clahe = cv2.createCLAHE(
            clipLimit=max(cfg.clahe_clip_limit, 0.1),
            tileGridSize=cfg.clahe_grid_size,
        )
        gray = clahe.apply(gray)
    return gray


def _rotate_gray(gray: np.ndarray, deg: int) -> np.ndarray:
    deg = deg % 360
    if deg == 0:
        return gray
    if deg == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(gray, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # General rotation using warpAffine (rare path)
    h, w = gray.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, float(deg), 1.0)
    rotated = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated


def _lane_mask(gray: np.ndarray, axis: int, cfg: FigcheckHeuristicConfig) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        cfg.adaptive_block_size,
        cfg.adaptive_c,
    )
    kernel_size = (cfg.lane_kernel, 1) if axis == 0 else (1, cfg.lane_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def _lane_profile(mask: np.ndarray, axis: int) -> np.ndarray:
    profile = mask.mean(axis=axis).astype(np.float32)
    if profile.size == 0:
        return profile
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 5), 0).flatten()
    if np.max(profile) > 0:
        profile = profile / np.max(profile)
    return profile


def _projection_alignment(arr_a: np.ndarray, arr_b: np.ndarray, cfg: FigcheckHeuristicConfig) -> float:
    a = np.asarray(arr_a, dtype=cfg.dtype)
    b = np.asarray(arr_b, dtype=cfg.dtype)
    if a.size == 0 or b.size == 0:
        return 0.0
    a = (a - np.mean(a)) / (np.std(a) + cfg.profile_eps)
    b = (b - np.mean(b)) / (np.std(b) + cfg.profile_eps)
    corr = np.correlate(a, b, mode="full")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + cfg.profile_eps
    value = float(np.max(corr) / denom)
    return float(np.clip(value, -1.0, 1.0))


def _lane_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.size == 0 or mask_b.size == 0:
        return 0.0
    h = min(mask_a.shape[0], mask_b.shape[0])
    w = min(mask_a.shape[1], mask_b.shape[1])
    if h == 0 or w == 0:
        return 0.0
    resized_a = cv2.resize(mask_a, (w, h), interpolation=cv2.INTER_NEAREST)
    resized_b = cv2.resize(mask_b, (w, h), interpolation=cv2.INTER_NEAREST)
    bin_a = resized_a > 0
    bin_b = resized_b > 0
    intersection = np.logical_and(bin_a, bin_b).sum()
    union = np.logical_or(bin_a, bin_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _partial_template_score(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    cfg: FigcheckHeuristicConfig,
) -> float:
    scores: List[float] = []
    for template, search in ((gray_a, gray_b), (gray_b, gray_a)):
        tpl = template.copy()
        src = search.copy()
        if tpl.shape[0] >= src.shape[0] or tpl.shape[1] >= src.shape[1]:
            scale_y = src.shape[0] / tpl.shape[0]
            scale_x = src.shape[1] / tpl.shape[1]
            scale = min(scale_x, scale_y)
            if scale <= 0:
                continue
            scale = max(scale, cfg.min_partial_scale)
            new_w = max(int(round(tpl.shape[1] * scale)), cfg.partial_template_min_size)
            new_h = max(int(round(tpl.shape[0] * scale)), cfg.partial_template_min_size)
            tpl = cv2.resize(tpl, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if tpl.shape[0] >= src.shape[0] or tpl.shape[1] >= src.shape[1]:
            continue
        if min(tpl.shape[:2]) < cfg.partial_template_min_size:
            continue
        result = cv2.matchTemplate(src, tpl, cv2.TM_CCOEFF_NORMED)
        if result.size == 0:
            continue
        scores.append(float(np.max(result)))
    return max(scores) if scores else 0.0

