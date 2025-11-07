"""Western blot lane normalization utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class LaneNormalizationResult:
    """Summary of lane detection and normalization for a panel."""

    is_candidate: bool
    texture_score: float
    rotation_angle: float
    lane_count: int
    lane_regions: List[Tuple[int, int]]
    lines: List[Tuple[int, int, int, int]]
    image_shape: Tuple[int, int]


def compute_vertical_texture_score(gray: np.ndarray) -> float:
    """Measure vertical frequency energy using the Sobel X gradient."""
    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    return float(np.mean(np.abs(sobel)))


def _detect_vertical_lines(
    gray: np.ndarray,
    canny_low: int = 30,
    canny_high: int = 100,
    min_length_ratio: float = 0.45,
    max_gap_ratio: float = 0.04,
) -> Tuple[List[Tuple[int, int, int, int]], float]:
    """Detect near-vertical lines using the probabilistic Hough transform."""
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    h, w = gray.shape[:2]
    diag = (h ** 2 + w ** 2) ** 0.5
    min_length = max(int(min(h, w) * min_length_ratio), 40)
    max_gap = max(int(diag * max_gap_ratio), 5)
    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(int(min(h, w) * 0.10), 20),
        minLineLength=min_length,
        maxLineGap=max_gap,
    )

    if lines_p is None:
        return [], 0.0

    vertical_lines: List[Tuple[int, int, int, int]] = []
    angles: List[float] = []

    for line in lines_p.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, line)
        dx = x2 - x1
        dy = y2 - y1
        if dy == 0:
            continue
        angle = np.degrees(np.arctan2(dx, dy))
        if abs(angle) <= 30.0:  # within ±30° of vertical
            vertical_lines.append((x1, y1, x2, y2))
            angles.append(angle)

    if not angles:
        return [], 0.0

    median_angle = float(np.median(angles))
    return vertical_lines, median_angle


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image, expanding the canvas to avoid cropping."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _extract_lane_regions(
    rotated_gray: np.ndarray,
    min_height_ratio: float = 0.6,
    min_width: int = 10,
) -> List[Tuple[int, int]]:
    """Identify vertical lane regions after rotation."""
    h, w = rotated_gray.shape[:2]
    blurred = cv2.GaussianBlur(rotated_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=8,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(int(h * 0.08), 15)))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lane_regions: List[Tuple[int, int]] = []
    min_height = int(h * min_height_ratio)

    for cnt in contours:
        x, y, width, height = cv2.boundingRect(cnt)
        if height < min_height or width < min_width:
            continue
        lane_regions.append((x, x + width))

    if not lane_regions:
        return []

    lane_regions = sorted(lane_regions, key=lambda lr: lr[0])

    merged: List[Tuple[int, int]] = []
    for region in lane_regions:
        if not merged:
            merged.append(region)
            continue
        prev_start, prev_end = merged[-1]
        start, end = region
        if start <= prev_end + 5:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append(region)

    return merged


def normalize_wb_panel(
    image_bgr: np.ndarray,
    texture_threshold: float = 12.0,
    min_vertical_lines: int = 1,
) -> Tuple[np.ndarray, LaneNormalizationResult]:
    """Deskew a potential Western blot panel to a canonical orientation."""
    if image_bgr is None:
        raise ValueError("image_bgr must be a valid image array")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr

    texture_score = compute_vertical_texture_score(gray)
    vertical_lines, median_angle = _detect_vertical_lines(gray)

    rotated = _rotate_image(image_bgr, -median_angle) if vertical_lines else image_bgr.copy()
    rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    lane_regions = _extract_lane_regions(rotated_gray)
    lane_count = len(lane_regions)

    is_candidate = (
        texture_score >= texture_threshold
        and len(vertical_lines) >= min_vertical_lines
        and lane_count >= 1
    )

    normalized = rotated if is_candidate else image_bgr.copy()

    result = LaneNormalizationResult(
        is_candidate=is_candidate,
        texture_score=texture_score,
        rotation_angle=-median_angle if vertical_lines else 0.0,
        lane_count=lane_count,
        lane_regions=[(int(a), int(b)) for a, b in lane_regions],
        lines=[tuple(map(int, line)) for line in vertical_lines],
        image_shape=normalized.shape[:2],
    )

    return normalized, result


def compute_lane_profiles(
    gray_image: np.ndarray,
    lane_regions: Sequence[Tuple[int, int]],
    sample_length: int = 256,
) -> List[np.ndarray]:
    """Compute 1D intensity profiles for each lane region."""
    if gray_image.ndim != 2:
        raise ValueError("gray_image must be a single-channel array")

    if not lane_regions:
        lane_regions = [(0, gray_image.shape[1])]

    profiles: List[np.ndarray] = []

    for start, end in lane_regions:
        start = max(0, min(gray_image.shape[1] - 1, int(start)))
        end = max(start + 1, min(gray_image.shape[1], int(end)))
        lane_slice = gray_image[:, start:end]
        column_profile = lane_slice.mean(axis=1).astype(np.float32)

        original_length = column_profile.shape[0]
        if original_length <= 1:
            continue

        x = np.linspace(0, original_length - 1, num=original_length, dtype=np.float32)
        xp = np.linspace(0, original_length - 1, num=sample_length, dtype=np.float32)
        resampled = np.interp(xp, x, column_profile)

        resampled = resampled.astype(np.float32)
        resampled -= resampled.min()
        denom = resampled.max()
        if denom > 1e-6:
            resampled /= denom

        profiles.append(resampled)

    return profiles


def dynamic_time_warping_distance(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """Compute a simple DTW distance between two 1D profiles."""
    len_a = profile_a.shape[0]
    len_b = profile_b.shape[0]

    cost = np.full((len_a + 1, len_b + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            diff = abs(profile_a[i - 1] - profile_b[j - 1])
            cost[i, j] = diff + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(cost[len_a, len_b] / (len_a + len_b))


def lane_profile_set_distance(
    profiles_a: Sequence[np.ndarray],
    profiles_b: Sequence[np.ndarray],
    unmatched_penalty: float = 1.0,
) -> Optional[float]:
    """Aggregate DTW distance across lane profile sets."""
    if not profiles_a or not profiles_b:
        return None

    count = min(len(profiles_a), len(profiles_b))
    distances: List[float] = []

    for idx in range(count):
        distances.append(dynamic_time_warping_distance(profiles_a[idx], profiles_b[idx]))

    if len(profiles_a) != len(profiles_b):
        distances.extend([unmatched_penalty] * abs(len(profiles_a) - len(profiles_b)))

    return float(np.mean(distances))

