"""Bangla handwritten-text line segmentation.

Pipeline:
1) Deskew lightly using Hough lines.
2) Bilateral-filter denoise and adaptive Gaussian binarize.
3) Estimate a robust per-image line height from connected-component statistics.
4) Smooth the horizontal projection with sigma = line_h / 4.
5) Find peaks (ink bands). Merge peaks that are closer than line_h * 0.5.
6) Compute line bounds at midpoints between consecutive peaks, trimmed to where
   the local projection stays above a robust threshold.
7) Final cleanup: drop lines shorter than ~line_h/3 unless only 1-2 lines remain.
"""
from __future__ import annotations
from typing import Optional, List, Tuple
import cv2
import numpy as np
import os


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _deskew(image_bgr: np.ndarray, gray: np.ndarray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 200,
        minLineLength=gray.shape[1] // 3, maxLineGap=20,
    )
    angle = 0.0
    if lines is not None and len(lines) > 5:
        lines = lines.reshape(-1, 4)
        angs = []
        for x1, y1, x2, y2 in lines:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < a < 45:
                angs.append(a)
        if angs:
            angle = float(np.median(angs))
    if abs(angle) > 0.3:
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        gray = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        image_bgr = cv2.warpAffine(
            image_bgr, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
    return image_bgr, gray, angle


def preprocess(image_bgr):
    """Return (image_bgr, gray, bw, deskew_angle)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_bgr, gray, angle = _deskew(image_bgr, gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 15,
    )
    return image_bgr, gray, bw, angle


def preprocess_path(path):
    img = cv2.imread(path)
    return preprocess(img)


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------

def _estimate_line_height(bw: np.ndarray) -> int:
    H, W = bw.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    heights = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 40:
            continue
        if h < 10 or h > H * 0.3:
            continue
        if w > W * 0.7 and h < H * 0.02:  # horizontal border line
            continue
        if w > 0 and h > 0 and (w / h) > 25:
            continue
        heights.append(h)
    if len(heights) < 10:
        return max(40, H // 12)
    heights.sort()
    char_h = heights[len(heights) // 2]
    return max(50, int(char_h * 2.2))


# ---------------------------------------------------------------------------
# Main segmentation routine
# ---------------------------------------------------------------------------

def _smooth_profile(p: np.ndarray, sigma: float) -> np.ndarray:
    k = int(sigma * 6) | 1
    return cv2.GaussianBlur(p.reshape(-1, 1), (1, k), 0).ravel()


def _find_peaks(profile: np.ndarray, min_height: float) -> list:
    peaks = []
    for i in range(2, len(profile) - 2):
        if (
            profile[i] > profile[i - 1]
            and profile[i] >= profile[i + 1]
            and profile[i] > min_height
        ):
            peaks.append(i)
    return peaks


def segment_lines(bw: np.ndarray, line_h: int | None = None):
    """Detect text-line bounding boxes from an inverted binarized image.

    Returns: (line_h, [(top, bottom, left, right), ...]) ordered top->bottom.
    """
    H, W = bw.shape
    if line_h is None:
        line_h = _estimate_line_height(bw)

    proj = bw.sum(axis=1) / 255.0
    sigma = max(2.0, line_h / 4.0)
    sp = _smooth_profile(proj, sigma)

    # Robust peak threshold: 10% of profile maximum, but at least 4 ink-pixels worth.
    peak_min = max(4.0, sp.max() * 0.10)
    peaks = _find_peaks(sp, peak_min)

    if not peaks:
        return line_h, []

    # Merge peaks separated by less than ~half a line height.
    merged = [peaks[0]]
    for p in peaks[1:]:
        if p - merged[-1] < line_h * 0.5:
            if sp[p] > sp[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)

    # Build per-line bounding boxes using midpoints between peaks.
    # The horizontal extent uses only the columns that contain ink in that y-range.
    boxes = []
    for i, p in enumerate(merged):
        top = 0 if i == 0 else (merged[i - 1] + p) // 2
        bot = H - 1 if i == len(merged) - 1 else (p + merged[i + 1]) // 2

        # Trim to where the smoothed profile exceeds 20% of the peak.
        peak_val = max(1.0, sp[p])
        thresh = max(2.0, peak_val * 0.20)
        while top < p and sp[top] < thresh:
            top += 1
        while bot > p and sp[bot] < thresh:
            bot -= 1

        # Compute horizontal extent using the original (un-smoothed) projection
        # per column restricted to this y-band.
        band = bw[top:bot + 1]
        col_sum = band.sum(axis=0) / 255.0
        nz = np.where(col_sum > 0)[0]
        if nz.size == 0:
            continue
        left = max(0, int(nz[0]) - 8)
        right = min(W - 1, int(nz[-1]) + 8)
        boxes.append((int(top), int(bot), int(left), int(right)))

    # Drop tiny lines (likely noise) unless very few remain.
    if len(boxes) > 3:
        boxes = [(t, b, l, r) for t, b, l, r in boxes
                 if b - t + 1 >= max(8, line_h // 3)]
    return line_h, boxes


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

def segment_path(path: str):
    img = cv2.imread(path)
    return segment_image(img)


def segment_image(image_bgr):
    """Full pipeline: takes a BGR image (or path), returns line boxes + debug info."""
    image_bgr, gray, bw, angle = preprocess(image_bgr)
    line_h, boxes = segment_lines(bw)
    return {
        "image": image_bgr,
        "gray": gray,
        "bw": bw,
        "deskew_angle": angle,
        "line_height": line_h,
        "boxes": boxes,            # list of (top, bot, left, right)
        "count": len(boxes),
    }


def draw_boxes(image_bgr, boxes, color=(0, 0, 255), thickness=4, labels=True):
    vis = image_bgr.copy()
    for i, (t, b, l, r) in enumerate(boxes, 1):
        cv2.rectangle(vis, (l, t), (r, b), color, thickness)
        if labels:
            cv2.putText(
                vis, str(i), (l + 10, max(t + 40, 50)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4,
            )
    return vis


def crop_lines(image_bgr, boxes, pad=8):
    """Crop each line out of the original image with padding."""
    h, w = image_bgr.shape[:2]
    crops = []
    pad_h = pad
    pad_w = pad + 6  # Extra horizontal padding to avoid clipping initial/trailing diacritics
    for t, b, l, r in boxes:
        t0 = max(0, t - pad_h)
        b0 = min(h - 1, b + pad_h)
        l0 = max(0, l - pad_w)
        r0 = min(w - 1, r + pad_w)
        crops.append(image_bgr[t0:b0 + 1, l0:r0 + 1])
    return crops
