import os
import glob
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# -----------------------------
# Utilities
# -----------------------------
@dataclass
class OCRLine:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    text: str


@dataclass
class OCRPage:
    file: str
    text: str
    lines: List[OCRLine]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


# -----------------------------
# Page detection + flattening
# -----------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """

    :param image: original BGR image
    :param pts: array of shape (4,2) with the corner points of the detected page contour
     in any order (will be ordered inside). Coordinates are in the original image space.
     The points should correspond to the corners of the page in the image.
     The order of points can be arbitrary; the function will sort them to top-left, top-right, bottom-right, bottom-left.
     The points should ideally form a quadrilateral that represents the page in the image. The function will then compute a perspective transform to "flatten" this quadrilateral
    :return:
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped


def find_page_and_warp(img_bgr: np.ndarray) -> np.ndarray:
    """
    Attempts to find the notebook page contour and perspective-warp it.
    If it fails, returns the original image.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)

    # Close gaps in edges
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * (img_bgr.shape[0] * img_bgr.shape[1]):
            pts = approx.reshape(4, 2).astype("float32")
            return four_point_transform(img_bgr, pts)

    return img_bgr


# -----------------------------
# Cleanup (shadow removal, contrast, binarize)
# -----------------------------
def preprocess_for_segmentation(img_bgr: np.ndarray) -> np.ndarray:
    """
    Returns a clean binary-ish image suitable for finding text lines.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Illumination normalization (helps with shadows)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    norm = cv2.divide(gray, bg, scale=255)

    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    norm = cv2.fastNlMeansDenoising(norm, h=12)

    # Adaptive threshold
    th = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Make "ink" white for easier morphology (optional, depends on threshold)
    # Here: black text on white background is typical, so invert to get text as white blobs:
    inv = 255 - th
    return inv


# -----------------------------
# Line segmentation
# -----------------------------
def segment_lines(inv_bin: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Given an inverted binary image (text ~white blobs), return bounding boxes for lines.
    Approach:
      - horizontal dilation to merge characters into line blobs
      - contour detection
      - filter + sort top-to-bottom
    """
    h, w = inv_bin.shape[:2]

    # Kernel width scales with page width. Bigger merges words into line.
    kW = max(25, w // 30)
    kH = max(3, h // 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kW, kH))

    merged = cv2.dilate(inv_bin, kernel, iterations=2)
    merged = cv2.erode(merged, kernel, iterations=1)

    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)

        # Heuristics: ignore tiny noise
        if bh < max(12, h // 120):
            continue
        if bw < max(80, w // 8):
            continue

        # Clamp boxes to image bounds
        x = max(0, x)
        y = max(0, y)
        bw = min(w - x, bw)
        bh = min(h - y, bh)
        boxes.append((x, y, bw, bh))

    # Sort top-to-bottom, then left-to-right
    boxes.sort(key=lambda b: (b[1], b[0]))

    # Optional: merge boxes that are extremely close vertically (handles broken lines)
    merged_boxes = []
    for b in boxes:
        if not merged_boxes:
            merged_boxes.append(b)
            continue
        x, y, bw, bh = b
        px, py, pbw, pbh = merged_boxes[-1]

        # If current line overlaps/near previous line vertically, merge
        if abs(y - py) < max(8, h // 300) and (x < px + pbw and px < x + bw):
            nx = min(px, x)
            ny = min(py, y)
            nxe = max(px + pbw, x + bw)
            nye = max(py + pbh, y + bh)
            merged_boxes[-1] = (nx, ny, nxe - nx, nye - ny)
        else:
            merged_boxes.append(b)

    return merged_boxes


def crop_with_padding(img: np.ndarray, box: Tuple[int, int, int, int], pad: int = 8) -> np.ndarray:
    x, y, w, h = box
    H, W = img.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return img[y1:y2, x1:x2]


# -----------------------------
# OCR (TrOCR)
# -----------------------------
class HandwritingOCR:
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten", max_length: int = 256):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.max_length = max_length

    def ocr_line(self, line_img_bgr_or_gray: np.ndarray) -> str:
        # TrOCR expects RGB PIL image
        if len(line_img_bgr_or_gray.shape) == 2:
            rgb = cv2.cvtColor(line_img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(line_img_bgr_or_gray, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb)
        pixel_values = self.processor(images=pil, return_tensors="pt").pixel_values
        ids = self.model.generate(pixel_values, max_length=self.max_length)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.strip()


def stitch_lines(lines: List[str]) -> str:
    """
    Basic stitching: join line texts with newlines.
    You can get fancier later (paragraph detection based on vertical gaps).
    """
    # Remove empty or low-signal lines
    clean = [ln for ln in (l.strip() for l in lines) if ln]
    return "\n".join(clean).strip() + ("\n" if clean else "")


# -----------------------------
# Pipeline entrypoints
# -----------------------------
def process_one_image(path: str, ocr: HandwritingOCR, debug_dir: Optional[str] = None) -> OCRPage:
    img = read_image_bgr(path)

    # 1) Detect page & warp (helps deskew/perspective)
    warped = find_page_and_warp(img)

    # 2) Preprocess for segmentation (inverted binary)
    inv = preprocess_for_segmentation(warped)

    # 3) Segment lines
    boxes = segment_lines(inv)

    if debug_dir:
        ensure_dir(debug_dir)
        vis = warped.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        dbg_path = os.path.join(debug_dir, os.path.splitext(os.path.basename(path))[0] + "_lines.jpg")
        cv2.imwrite(dbg_path, vis)

    # 4) OCR each line crop (use the cleaned grayscale-ish crop for OCR)
    lines_out: List[OCRLine] = []
    texts: List[str] = []

    # For OCR input, use a cleaner grayscale image (not inverted binary blobs)
    # We'll reuse a normalized grayscale for OCR:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    for box in boxes:
        crop = crop_with_padding(gray, box, pad=10)

        # Light cleanup on the crop: normalize + threshold (optional but often helps)
        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
        crop_th = cv2.adaptiveThreshold(
            crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )

        text = ocr.ocr_line(crop_th)
        lines_out.append(OCRLine(bbox=box, text=text))
        texts.append(text)

    full_text = stitch_lines(texts)

    return OCRPage(
        file=os.path.basename(path),
        text=full_text,
        lines=lines_out
    )


def process_folder(
    input_dir: str,
    output_dir: str,
    model_name: str = "microsoft/trocr-base-handwritten",
    limit: Optional[int] = None,
    debug: bool = True,
) -> List[OCRPage]:
    ensure_dir(output_dir)
    debug_dir = os.path.join(output_dir, "_debug") if debug else None

    exts = ["jpg", "jpeg", "png", "webp", "tif", "tiff"]
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(input_dir, f"*.{ext}"))
        files += glob.glob(os.path.join(input_dir, f"*.{ext.upper()}"))

    files = sorted(files)
    if limit is not None:
        files = files[:limit]

    ocr = HandwritingOCR(model_name=model_name)

    pages: List[OCRPage] = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.basename(path)}")
        page = process_one_image(path, ocr, debug_dir=debug_dir)

        base = os.path.splitext(os.path.basename(path))[0]
        txt_path = os.path.join(output_dir, f"{base}.txt")
        json_path = os.path.join(output_dir, f"{base}.json")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(page.text)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "file": page.file,
                    "text": page.text,
                    "lines": [{"bbox": l.bbox, "text": l.text} for l in page.lines],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        pages.append(page)

    # Write a batch index file for convenience
    index_path = os.path.join(output_dir, "results_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"file": p.file, "text_file": os.path.splitext(p.file)[0] + ".txt"} for p in pages],
            f,
            indent=2,
            ensure_ascii=False,
        )

    return pages


if __name__ == "__main__":
    INPUT_DIR = "./input_images"
    OUTPUT_DIR = "./output_text"

    process_folder(INPUT_DIR, OUTPUT_DIR, limit=None, debug=True)
                                         