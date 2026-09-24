"""
Scan a receipt or document photo, then optionally extract fields with Gemini.

Usage:
    python scan_document.py photo.jpeg
    python scan_document.py photo.jpeg --extract
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order corners as top-left, top-right, bottom-right, bottom-left.

    The cloned project's x-sort version fails on tall, thin receipts: the two
    leftmost points are often both at the bottom. Sum/diff ordering is stable.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array(
        [
            pts[np.argmin(total)],
            pts[np.argmin(diff)],
            pts[np.argmax(total)],
            pts[np.argmax(diff)],
        ],
        dtype=np.float32,
    )


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = max(int(width_a), int(width_b), 1)
    max_height = max(int(height_a), int(height_b), 1)
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = height / float(h)
    return cv2.resize(image, (int(w * scale), height), interpolation=cv2.INTER_AREA)


VALID_FORMATS = {".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"}
DEFAULT_IMAGE = ROOT / "sample.jpeg"
OUTPUT_DIR = ROOT / "output"
RESCALED_HEIGHT = 800.0


def collect_images(image: Path | None, images_dir: Path | None) -> list[Path]:
    if image is not None:
        path = image if image.is_absolute() else ROOT / image
        if not path.is_file():
            raise SystemExit(f"Image not found: {path}")
        return [path]

    if images_dir is not None:
        folder = images_dir if images_dir.is_absolute() else ROOT / images_dir
        if not folder.is_dir():
            raise SystemExit(f"Directory not found: {folder}")
        found = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in VALID_FORMATS
        )
        if not found:
            raise SystemExit(f"No images found in {folder}")
        return found

    if DEFAULT_IMAGE.is_file():
        return [DEFAULT_IMAGE]

    raise SystemExit(
        "No image provided and sample.jpeg was not found in the project root."
    )


def _quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def _valid_quad(quad: np.ndarray, width: int, height: int) -> bool:
    area = cv2.contourArea(quad.astype(np.float32))
    image_area = float(width * height)
    if area < image_area * 0.08 or area > image_area * 0.985:
        return False

    ordered = order_points(quad)
    (tl, tr, br, bl) = ordered
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if w < 20 or h < 20:
        return False

    # Reject thin strips (e.g. table edge) that upscale into huge Telegram photos.
    if w < width * 0.2 or h < height * 0.2:
        return False

    # Reject near-square table crops; receipts and pages are usually taller.
    aspect = max(w, h) / min(w, h)
    if aspect > 10:
        return False
    return aspect >= 1.02


def _score_quad(quad: np.ndarray, width: int, height: int) -> float:
    area = cv2.contourArea(quad.astype(np.float32))
    ordered = order_points(quad)
    (tl, tr, br, bl) = ordered
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    aspect = max(w, h) / max(min(w, h), 1.0)
    # Prefer the whole page. A 22% target was locking onto one table on the page.
    coverage = area / float(width * height)
    return coverage * 4.0 + min(aspect, 3.0) * 0.15


def detect_from_text_blob(image: np.ndarray) -> np.ndarray | None:
    """Merge printed text into one blob — works well for receipts on a table."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 35))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), 1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    height, width = image.shape[:2]
    best = None
    best_score = -1.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        quad = _quad_from_contour(contour)
        if quad is None or not _valid_quad(quad, width, height):
            continue
        # Pad a little so empty paper around the text is kept.
        ordered = order_points(quad)
        top = ordered[1] - ordered[0]
        left = ordered[3] - ordered[0]
        pad_x = 0.08 * np.linalg.norm(top)
        pad_y = 0.06 * np.linalg.norm(left)
        ux = top / max(np.linalg.norm(top), 1.0)
        uy = left / max(np.linalg.norm(left), 1.0)
        padded = np.array(
            [
                ordered[0] - ux * pad_x - uy * pad_y,
                ordered[1] + ux * pad_x - uy * pad_y,
                ordered[2] + ux * pad_x + uy * pad_y,
                ordered[3] - ux * pad_x + uy * pad_y,
            ],
            dtype=np.float32,
        )
        padded[:, 0] = np.clip(padded[:, 0], 0, width - 1)
        padded[:, 1] = np.clip(padded[:, 1], 0, height - 1)
        score = _score_quad(padded, width, height)
        if score > best_score:
            best_score = score
            best = padded
    return best


def detect_from_paper_edges(image: np.ndarray) -> np.ndarray | None:
    """Find the paper outline from faint shadows on a light table."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 20, 70)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 1)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    )

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    height, width = image.shape[:2]
    best = None
    best_score = -1.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        quad = _quad_from_contour(contour)
        if quad is None or not _valid_quad(quad, width, height):
            continue
        score = _score_quad(quad, width, height)
        if score > best_score:
            best_score = score
            best = quad
    return best


def _quad_from_box(x: int, y: int, box_w: int, box_h: int, width: int, height: int, pad: float) -> np.ndarray:
    pad_x = int(box_w * pad)
    pad_y = int(box_h * pad)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width - 1, x + box_w + pad_x)
    y1 = min(height - 1, y + box_h + pad_y)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _content_quad(image: np.ndarray) -> np.ndarray | None:
    """Crop to the white sheet. On a full-page photo, trim the empty margin."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    _, paper = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    paper = cv2.morphologyEx(
        paper,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)),
        iterations=2,
    )
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, box_w, box_h = cv2.boundingRect(contour)
        coverage = (box_w * box_h) / float(width * height)
        if 0.08 <= coverage <= 0.92 and box_w > width * 0.2 and box_h > height * 0.2:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)
            return box

    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = cv2.findNonZero(ink)
    if points is None:
        return None
    x, y, box_w, box_h = cv2.boundingRect(points)
    return _quad_from_box(x, y, box_w, box_h, width, height, 0.03)


def _quads_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Turn a paper mask into four-corner frames, including skewed pages."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads: list[np.ndarray] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(contour, True)
        if peri < 40:
            continue
        for factor in (0.01, 0.02, 0.035, 0.05):
            approx = cv2.approxPolyDP(contour, factor * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quads.append(approx.reshape(4, 2).astype(np.float32))
                break
        else:
            rect = cv2.minAreaRect(contour)
            quads.append(cv2.boxPoints(rect).astype(np.float32))
    return quads


def _paper_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 1)
    closed = cv2.morphologyEx(
        cv2.bitwise_or(bright, edges),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        iterations=2,
    )
    return closed


def detect_document_frame(image: np.ndarray) -> np.ndarray | None:
    """Find the page corners the way a scanner does, even when the page is tilted."""
    height, width = image.shape[:2]
    mask = _paper_mask(image)
    best = None
    best_score = -1.0
    for quad in _quads_from_mask(mask):
        if not _valid_quad(quad, width, height):
            continue
        area = cv2.contourArea(quad.astype(np.float32))
        ordered = order_points(quad)
        angles = 0.0
        for index in range(4):
            pivot = ordered[index]
            before = ordered[index - 1] - pivot
            after = ordered[(index + 1) % 4] - pivot
            denom = float(np.linalg.norm(before) * np.linalg.norm(after)) + 1e-6
            angles += 1.0 - abs(float(np.dot(before, after)) / denom)
        score = (area / float(width * height)) * 5.0 + angles
        if score > best_score:
            best_score = score
            best = ordered
    return best


def _text_is_sideways(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = resize_to_height(gray, 400) if gray.shape[0] > 400 else gray
    horizontal = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    vertical = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    return float(np.mean(np.abs(vertical))) > float(np.mean(np.abs(horizontal))) * 1.2


def make_upright(image: np.ndarray) -> np.ndarray:
    """Rotate a straightened page so the text lines run left to right."""
    if not _text_is_sideways(image):
        return image
    turned = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if _text_is_sideways(turned):
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return turned


def find_document_quad(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    frame = detect_document_frame(image)
    if frame is not None:
        return frame
    content = _content_quad(image)
    if content is not None:
        coverage = cv2.contourArea(content.astype(np.float32)) / float(width * height)
        # A sheet on a table is the receipt. Do not replace it with a bigger outline.
        if coverage <= 0.92:
            return content
    content_area = (
        cv2.contourArea(content.astype(np.float32))
        if content is not None
        else float(width * height)
    )
    candidates: list[np.ndarray] = []
    for detector in (
        detect_from_text_blob,
        detect_from_paper_edges,
    ):
        quad = detector(image)
        if quad is None:
            continue
        # A payment table or one paragraph is not the receipt.
        if cv2.contourArea(quad.astype(np.float32)) < content_area * 0.65:
            continue
        candidates.append(quad)
    if content is not None:
        candidates.append(content)

    if not candidates:
        return np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )

    return max(candidates, key=lambda q: _score_quad(q, width, height))


# Telegram sendPhoto: width + height <= 10_000 and aspect ratio <= 20:1.
TELEGRAM_MAX_PHOTO_SUM = 10_000
TELEGRAM_MAX_PHOTO_RATIO = 20.0
# Stay slightly inside Telegram limits to avoid rounding rejections.
TELEGRAM_MAX_PHOTO_SUM_SAFE = 9_980
TELEGRAM_MAX_PHOTO_RATIO_SAFE = 19.9


def _photo_limits_ok(width: int, height: int) -> bool:
    if width < 1 or height < 1:
        return False
    if width + height > TELEGRAM_MAX_PHOTO_SUM_SAFE:
        return False
    return max(width, height) / min(width, height) <= TELEGRAM_MAX_PHOTO_RATIO_SAFE


def fit_telegram_photo(image: np.ndarray) -> np.ndarray:
    """Pad/resize so Telegram sendPhoto accepts the image."""
    pad_value = 255 if image.ndim == 2 else (255, 255, 255)

    for _ in range(8):
        height, width = image.shape[:2]
        if _photo_limits_ok(width, height):
            return image

        ratio = max(width, height) / min(width, height)
        if ratio > TELEGRAM_MAX_PHOTO_RATIO_SAFE:
            if height > width:
                new_width = int(np.ceil(height / TELEGRAM_MAX_PHOTO_RATIO_SAFE))
                pad = new_width - width
                image = cv2.copyMakeBorder(
                    image, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=pad_value
                )
            else:
                new_height = int(np.ceil(width / TELEGRAM_MAX_PHOTO_RATIO_SAFE))
                pad = new_height - height
                image = cv2.copyMakeBorder(
                    image, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_CONSTANT, value=pad_value
                )
            continue

        if width + height > TELEGRAM_MAX_PHOTO_SUM_SAFE:
            scale = TELEGRAM_MAX_PHOTO_SUM_SAFE / float(width + height)
            image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            continue

        break

    return image


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])


def ensure_telegram_photo_file(path: Path) -> Path:
    """Resize only when Telegram would reject the photo. Leave good files untouched."""
    image = cv2.imread(str(path))
    if image is None:
        return path
    height, width = image.shape[:2]
    if _photo_limits_ok(width, height):
        return path
    _write_jpeg(path, fit_telegram_photo(image))
    return path


def _upscale_page(image: np.ndarray, target_width: int = 900) -> np.ndarray:
    height, width = image.shape[:2]
    if width >= target_width:
        return image
    scale = target_width / width
    return cv2.resize(
        image,
        (target_width, max(1, int(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


def enhance_readable(warped: np.ndarray) -> np.ndarray:
    """Keep the scan at its real size. Upscaling only softens text."""
    return warped


def enhance_high_contrast(warped: np.ndarray) -> np.ndarray:
    """Original Document-Scanner look: hard black-and-white threshold."""
    warped = _upscale_page(warped)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
    block = max(15, (sharpen.shape[1] // 40) | 1)
    return cv2.adaptiveThreshold(
        sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 12
    )


def _usable_page(image: np.ndarray) -> bool:
    height, width = image.shape[:2]
    if width < 80 or height < 80:
        return False
    aspect = max(width, height) / min(width, height)
    return aspect <= 12


def scan_image(image_path: Path, high_contrast: bool = False, output_path: Path | None = None) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    ratio = image.shape[0] / RESCALED_HEIGHT
    rescaled = resize_to_height(image, int(RESCALED_HEIGHT))
    quad = find_document_quad(rescaled)
    quad_area = cv2.contourArea(quad.astype(np.float32))
    frame_area = float(rescaled.shape[0] * rescaled.shape[1])
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / image_path.name
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = order_points(quad)
    corners = np.array(
        [[0, 0], [rescaled.shape[1] - 1, 0], [rescaled.shape[1] - 1, rescaled.shape[0] - 1], [0, rescaled.shape[0] - 1]],
        dtype=np.float32,
    )
    already_framed = quad_area >= frame_area * 0.97 and float(np.max(np.abs(ordered - corners))) < 8
    if already_framed:
        if output_path.resolve() != image_path.resolve():
            shutil.copyfile(image_path, output_path)
        return output_path
    warped = four_point_transform(image, quad * ratio)
    warped = make_upright(warped)
    if not _usable_page(warped):
        warped = image.copy()
    scanned = enhance_high_contrast(warped) if high_contrast else enhance_readable(warped)
    scanned = fit_telegram_photo(scanned)

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / image_path.name
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jpeg(output_path, scanned)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan documents with sangamprashant/Document-Scanner."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "image_pos",
        nargs="?",
        type=Path,
        help="Path to a single image (defaults to sample.jpeg)",
    )
    group.add_argument("--image", type=Path, help="Path to a single image")
    group.add_argument("--images", type=Path, help="Directory of images to scan")
    parser.add_argument(
        "--bw",
        action="store_true",
        help="Use the original high-contrast black-and-white threshold",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Use Gemini to extract date, category, and amount from the scan",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = args.image or args.image_pos
    paths = collect_images(image, args.images)
    os.chdir(ROOT)
    for path in paths:
        print(f"Scanning {path} ...")
        result = scan_image(path, high_contrast=args.bw)
        print(f"Saved {result}")
        if args.extract:
            from extract_receipt import extract_receipt, print_receipt, save_receipt

            info = extract_receipt(result)
            json_path = save_receipt(info, result)
            print_receipt(info)
            print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
