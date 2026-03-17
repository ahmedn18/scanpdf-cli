#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ENHANCEMENT_PROFILES = {
    "soft": {"clip": 1.6, "alpha": 1.05, "beta": 6, "sharp": 0.25, "sigma": 1.0},
    "balanced": {"clip": 2.2, "alpha": 1.12, "beta": 12, "sharp": 0.55, "sigma": 1.2},
    "strong": {"clip": 3.0, "alpha": 1.20, "beta": 18, "sharp": 0.80, "sigma": 1.4},
}

BRIGHTNESS_LEVELS = {
    "dark": -14,
    "normal": 0,
    "bright": 14,
    "very-bright": 26,
}


@dataclass(frozen=True)
class PreprocessConfig:
    inner_crop: float = 0.02
    min_page_area_ratio: float = 0.55
    deskew_min_angle: float = 0.5
    deskew_max_angle: float = 20.0
    enhance_profile: str = "balanced"
    brightness_level: str = "normal"
    paper_whiten: float = 0.35
    grayscale: bool = False
    png_compression: int = 3
    jpeg_quality: int = 60
    debug_dir: str | None = None
    debug_prefix: str = "debug"


def _validate_config(config: PreprocessConfig) -> None:
    if not 0.0 <= config.inner_crop < 0.2:
        raise ValueError("inner_crop must be in [0.0, 0.2)")
    if not 0.1 <= config.min_page_area_ratio <= 0.95:
        raise ValueError("min_page_area_ratio must be in [0.1, 0.95]")
    if config.deskew_min_angle < 0:
        raise ValueError("deskew_min_angle must be >= 0")
    if config.deskew_max_angle <= config.deskew_min_angle:
        raise ValueError("deskew_max_angle must be greater than deskew_min_angle")
    if config.enhance_profile not in ENHANCEMENT_PROFILES:
        raise ValueError(f"Unknown enhance_profile: {config.enhance_profile}")
    if config.brightness_level not in BRIGHTNESS_LEVELS:
        raise ValueError(f"Unknown brightness_level: {config.brightness_level}")
    if not 0 <= config.png_compression <= 9:
        raise ValueError("png_compression must be in [0, 9]")
    if not 1 <= config.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    if not 0.0 <= config.paper_whiten <= 1.0:
        raise ValueError("paper_whiten must be in [0.0, 1.0]")


def save_debug(debug_dir: str | None, prefix: str, name: str, image: np.ndarray) -> None:
    if not debug_dir:
        return
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(debug_dir) / f"{prefix}_{name}.png"), image)


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.hypot(br[0] - bl[0], br[1] - bl[1])
    width_b = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    max_width = max(int(width_a), int(width_b), 2)

    height_a = np.hypot(tr[0] - br[0], tr[1] - br[1])
    height_b = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    max_height = max(int(height_a), int(height_b), 2)

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def line_from_segment(seg: np.ndarray) -> tuple[float, float, float] | None:
    x1, y1, x2, y2 = [float(v) for v in seg]
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = np.hypot(a, b)
    if norm == 0:
        return None
    return (a / norm, b / norm, c / norm)


def intersect_lines(
    line1: tuple[float, float, float] | None, line2: tuple[float, float, float] | None
) -> list[float] | None:
    if line1 is None or line2 is None:
        return None
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return [x, y]


def is_valid_quad(
    quad: np.ndarray, width: int, height: int, min_area_ratio: float, max_gap_ratio: float
) -> bool:
    ordered = order_points(quad.astype("float32"))
    area = cv2.contourArea(ordered)
    if area < width * height * min_area_ratio:
        return False

    min_x = float(np.min(ordered[:, 0]))
    max_x = float(np.max(ordered[:, 0]))
    min_y = float(np.min(ordered[:, 1]))
    max_y = float(np.max(ordered[:, 1]))

    if min_x < -0.1 * width or min_y < -0.1 * height or max_x > 1.1 * width or max_y > 1.1 * height:
        return False

    gaps = [min_x / width, min_y / height, (width - max_x) / width, (height - max_y) / height]
    return all(g <= max_gap_ratio for g in gaps)


def detect_quad_hough(
    edged: np.ndarray, min_area_ratio: float, debug_vis: np.ndarray | None = None
) -> np.ndarray | None:
    h, w = edged.shape[:2]
    lines = cv2.HoughLinesP(
        edged,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=int(0.25 * max(w, h)),
        maxLineGap=int(0.03 * max(w, h)),
    )
    if lines is None:
        return None

    candidates: dict[str, list[tuple[float, np.ndarray]]] = {"top": [], "bottom": [], "left": [], "right": []}
    for raw_line in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in raw_line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 0.25 * max(w, h):
            continue

        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle > 90:
            angle = 180 - angle

        if debug_vis is not None:
            cv2.line(debug_vis, (x1, y1), (x2, y2), (40, 40, 255), 1)

        if angle <= 15:
            y_mid = (y1 + y2) / 2.0
            if y_mid < h * 0.45:
                candidates["top"].append((length - 1.5 * y_mid, raw_line))
            if y_mid > h * 0.55:
                candidates["bottom"].append((length - 1.5 * (h - y_mid), raw_line))
        elif angle >= 75:
            x_mid = (x1 + x2) / 2.0
            if x_mid < w * 0.45:
                candidates["left"].append((length - 1.5 * x_mid, raw_line))
            if x_mid > w * 0.55:
                candidates["right"].append((length - 1.5 * (w - x_mid), raw_line))

    if any(not candidates[k] for k in candidates):
        return None

    top = max(candidates["top"], key=lambda x: x[0])[1]
    bottom = max(candidates["bottom"], key=lambda x: x[0])[1]
    left = max(candidates["left"], key=lambda x: x[0])[1]
    right = max(candidates["right"], key=lambda x: x[0])[1]

    if debug_vis is not None:
        for seg in (top, bottom, left, right):
            x1, y1, x2, y2 = [int(v) for v in seg]
            cv2.line(debug_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    tl = intersect_lines(line_from_segment(top), line_from_segment(left))
    tr = intersect_lines(line_from_segment(top), line_from_segment(right))
    br = intersect_lines(line_from_segment(bottom), line_from_segment(right))
    bl = intersect_lines(line_from_segment(bottom), line_from_segment(left))
    if any(p is None for p in (tl, tr, br, bl)):
        return None

    quad = np.array([tl, tr, br, bl], dtype="float32")
    if not is_valid_quad(quad, w, h, min_area_ratio=min_area_ratio, max_gap_ratio=0.22):
        return None
    return order_points(quad)


def detect_quad_contour(edged: np.ndarray, min_area_ratio: float) -> np.ndarray | None:
    h, w = edged.shape[:2]
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

    best_quad = None
    best_score = -1.0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = approx.reshape(4, 2).astype("float32")
        if not is_valid_quad(quad, w, h, min_area_ratio=min_area_ratio, max_gap_ratio=0.35):
            continue

        ordered = order_points(quad)
        area_ratio = cv2.contourArea(ordered) / float(w * h)
        min_x = float(np.min(ordered[:, 0]))
        max_x = float(np.max(ordered[:, 0]))
        min_y = float(np.min(ordered[:, 1]))
        max_y = float(np.max(ordered[:, 1]))
        edge_gaps = np.array([min_x / w, min_y / h, (w - max_x) / w, (h - max_y) / h])
        edge_score = 1.0 - np.clip(edge_gaps.mean(), 0.0, 1.0)
        score = 0.75 * area_ratio + 0.25 * edge_score

        if score > best_score:
            best_score = score
            best_quad = ordered
    return best_quad


def crop_inner_margin(image: np.ndarray, ratio: float) -> np.ndarray:
    if ratio <= 0:
        return image
    h, w = image.shape[:2]
    mx = int(w * ratio)
    my = int(h * ratio)
    if mx <= 0 or my <= 0:
        return image
    if (mx * 2) >= w or (my * 2) >= h:
        return image
    return image[my : h - my, mx : w - mx]


def detect_and_warp(
    image: np.ndarray,
    min_page_area_ratio: float,
    inner_crop: float,
    debug_dir: str | None = None,
    debug_prefix: str = "debug",
) -> np.ndarray:
    orig = image.copy()
    h, w = image.shape[:2]

    target_max_dim = 1400
    scale = min(1.0, target_max_dim / float(max(h, w)))
    small = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    save_debug(debug_dir, debug_prefix, "01_gray", gray)
    save_debug(debug_dir, debug_prefix, "02_edges", edges)

    debug_hough = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if debug_dir else None
    hough_quad = detect_quad_hough(edges, min_area_ratio=min_page_area_ratio, debug_vis=debug_hough)
    if debug_hough is not None:
        save_debug(debug_dir, debug_prefix, "03_hough_lines", debug_hough)

    if hough_quad is not None:
        quad = hough_quad
    else:
        contour_quad = detect_quad_contour(edges, min_area_ratio=min_page_area_ratio)
        if contour_quad is None:
            return orig
        quad = contour_quad

    preview = small.copy()
    cv2.polylines(preview, [quad.astype("int32")], True, (0, 255, 0), 2)
    save_debug(debug_dir, debug_prefix, "04_selected_quad", preview)

    if scale < 1.0:
        quad = quad / scale
    warped = four_point_transform(orig, quad)
    save_debug(debug_dir, debug_prefix, "05_warped", warped)
    cropped = crop_inner_margin(warped, inner_crop)
    save_debug(debug_dir, debug_prefix, "06_cropped", cropped)
    return cropped


def deskew(
    image: np.ndarray,
    min_angle: float,
    max_angle: float,
    debug_dir: str | None = None,
    debug_prefix: str = "debug",
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    abs_angle = abs(angle)
    if abs_angle < min_angle or abs_angle > max_angle:
        return image

    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    save_debug(debug_dir, debug_prefix, "07_deskewed", rotated)
    return rotated


def enhance_image(
    image: np.ndarray,
    profile_name: str,
    brightness_level: str,
    paper_whiten: float,
    debug_dir: str | None = None,
    debug_prefix: str = "debug",
) -> np.ndarray:
    profile = ENHANCEMENT_PROFILES[profile_name]
    brightness_shift = BRIGHTNESS_LEVELS[brightness_level]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=profile["clip"], tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    l_channel = cv2.convertScaleAbs(l_channel, alpha=profile["alpha"], beta=profile["beta"])
    l_channel = cv2.convertScaleAbs(l_channel, alpha=1.0, beta=brightness_shift)

    if paper_whiten > 0:
        background = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=31, sigmaY=31)
        background = np.maximum(background, 1)
        normalized = cv2.divide(l_channel, background, scale=255)
        l_channel = cv2.addWeighted(l_channel, 1.0 - paper_whiten, normalized, paper_whiten, 0)
        l_channel = cv2.convertScaleAbs(l_channel, alpha=1.0 + (0.06 * paper_whiten), beta=18 * paper_whiten)

    if profile["sharp"] > 0:
        blurred = cv2.GaussianBlur(l_channel, (0, 0), profile["sigma"])
        l_channel = cv2.addWeighted(l_channel, 1.0 + profile["sharp"], blurred, -profile["sharp"], 0)

    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    save_debug(debug_dir, debug_prefix, "08_enhanced", enhanced)
    return enhanced


def write_output(path: str, image: np.ndarray, png_compression: int, jpeg_quality: int) -> bool:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
    return cv2.imwrite(path, image, params)


def process_image(input_path: str, output_path: str, config: PreprocessConfig) -> None:
    _validate_config(config)
    image = cv2.imread(input_path)
    if image is None:
        raise RuntimeError(f"Cannot read: {input_path}")

    image = detect_and_warp(
        image,
        min_page_area_ratio=config.min_page_area_ratio,
        inner_crop=config.inner_crop,
        debug_dir=config.debug_dir,
        debug_prefix=config.debug_prefix,
    )
    image = deskew(
        image,
        min_angle=config.deskew_min_angle,
        max_angle=config.deskew_max_angle,
        debug_dir=config.debug_dir,
        debug_prefix=config.debug_prefix,
    )
    image = enhance_image(
        image,
        config.enhance_profile,
        config.brightness_level,
        config.paper_whiten,
        debug_dir=config.debug_dir,
        debug_prefix=config.debug_prefix,
    )

    if config.grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        save_debug(config.debug_dir, config.debug_prefix, "09_grayscale", image)

    ok = write_output(output_path, image, config.png_compression, config.jpeg_quality)
    if not ok:
        raise RuntimeError(f"Failed to write output: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect page, perspective-correct, deskew, and enhance scanned images."
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--inner-crop", type=float, default=0.02, help="Inner crop ratio after warp (default: 0.02)")
    parser.add_argument(
        "--min-page-area-ratio",
        type=float,
        default=0.55,
        help="Minimum detected page area as fraction of image area (default: 0.55)",
    )
    parser.add_argument("--deskew-min-angle", type=float, default=0.5, help="Minimum deskew angle to apply (default: 0.5)")
    parser.add_argument("--deskew-max-angle", type=float, default=20.0, help="Maximum deskew angle to apply (default: 20)")
    parser.add_argument(
        "--enhance-profile",
        choices=sorted(ENHANCEMENT_PROFILES.keys()),
        default="balanced",
        help="Enhancement profile (default: balanced)",
    )
    parser.add_argument(
        "--brightness-level",
        choices=sorted(BRIGHTNESS_LEVELS.keys()),
        default="normal",
        help="Brightness level for final output (default: normal)",
    )
    parser.add_argument(
        "--paper-whiten",
        type=float,
        default=0.35,
        help="Background whitening strength 0.0-1.0 (default: 0.35)",
    )
    parser.add_argument("--grayscale", action="store_true", help="Convert final output to grayscale")
    parser.add_argument(
        "--png-compression",
        type=int,
        default=3,
        help="PNG compression level 0-9 (lower is faster, default: 3)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=60,
        help="JPEG quality 1-100 for .jpg output (default: 60)",
    )
    parser.add_argument("--debug-dir", help="Directory to save intermediate debug images")
    parser.add_argument("--debug-prefix", default="debug", help="Prefix for debug output filenames")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = PreprocessConfig(
        inner_crop=args.inner_crop,
        min_page_area_ratio=args.min_page_area_ratio,
        deskew_min_angle=args.deskew_min_angle,
        deskew_max_angle=args.deskew_max_angle,
        enhance_profile=args.enhance_profile,
        brightness_level=args.brightness_level,
        paper_whiten=args.paper_whiten,
        grayscale=args.grayscale,
        png_compression=args.png_compression,
        jpeg_quality=args.jpeg_quality,
        debug_dir=args.debug_dir,
        debug_prefix=args.debug_prefix,
    )
    try:
        process_image(args.input, args.output, config)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
