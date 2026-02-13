import os
import io
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image

from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# -----------------------------
# Data
# -----------------------------
@dataclass
class OCRLine:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    text: str


@dataclass
class OCRPage:
    filename: str
    text: str
    lines: List[OCRLine]


# -----------------------------
# Page warp helpers
# -----------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # tl
    rect[2] = pts[np.argmax(s)]       # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # tr
    rect[3] = pts[np.argmax(diff)]    # bl
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def find_page_and_warp(img_bgr: np.ndarray) -> np.ndarray:
    """
    Try to find a 4-corner page contour and warp.
    If not found, return original image.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    img_area = img_bgr.shape[0] * img_bgr.shape[1]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * img_area:
            pts = approx.reshape(4, 2).astype("float32")
            return _four_point_transform(img_bgr, pts)

    return img_bgr


# -----------------------------
# Preprocess + segmentation
# -----------------------------
def preprocess_for_segmentation(img_bgr: np.ndarray) -> np.ndarray:
    """
    Make an inverted binary-ish image for line blob detection (text becomes white blobs).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Illumination normalization (shadow reduction)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    norm = cv2.divide(gray, bg, scale=255)

    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    norm = cv2.fastNlMeansDenoising(norm, h=12)

    th = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    inv = 255 - th
    return inv


def segment_lines(inv_bin: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Horizontal dilation merges characters into line blobs.
    Then find contours -> bounding boxes sorted top-to-bottom.
    """
    h, w = inv_bin.shape[:2]

    kW = max(25, w // 30)
    kH = max(3, h // 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kW, kH))

    merged = cv2.dilate(inv_bin, kernel, iterations=2)
    merged = cv2.erode(merged, kernel, iterations=1)

    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh < max(12, h // 120):
            continue
        if bw < max(80, w // 8):
            continue
        boxes.append((x, y, bw, bh))

    boxes.sort(key=lambda b: (b[1], b[0]))

    # Merge “broken” line blobs that are very close
    merged_boxes: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        if not merged_boxes:
            merged_boxes.append(b)
            continue
        x, y, bw, bh = b
        px, py, pbw, pbh = merged_boxes[-1]

        if abs(y - py) < max(8, h // 300) and (x < px + pbw and px < x + bw):
            nx = min(px, x)
            ny = min(py, y)
            nxe = max(px + pbw, x + bw)
            nye = max(py + pbh, y + bh)
            merged_boxes[-1] = (nx, ny, nxe - nx, nye - ny)
        else:
            merged_boxes.append(b)

    return merged_boxes


def crop_with_padding(img: np.ndarray, box: Tuple[int, int, int, int], pad: int = 10) -> np.ndarray:
    x, y, w, h = box
    H, W = img.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return img[y1:y2, x1:x2]


def stitch_lines_with_paragraphs(texts: List[str], boxes: List[Tuple[int, int, int, int]]) -> str:
    """
    Simple paragraph reconstruction based on vertical gaps between line boxes.
    If gap is larger than typical, insert a blank line.
    """
    cleaned = [(t.strip(), b) for t, b in zip(texts, boxes) if t and t.strip()]
    if not cleaned:
        return ""

    # Compute typical gap (median of successive gaps)
    ys = [b[1] for _, b in cleaned]
    hs = [b[3] for _, b in cleaned]
    gaps = []
    for i in range(1, len(cleaned)):
        prev_y, prev_h = cleaned[i - 1][1][1], cleaned[i - 1][1][3]
        curr_y = cleaned[i][1][1]
        gaps.append(max(0, curr_y - (prev_y + prev_h)))

    median_gap = int(np.median(gaps)) if gaps else 0
    # Threshold: “big gap” means new paragraph
    para_gap = max(18, median_gap * 2 + 6)

    out_lines: List[str] = []
    out_lines.append(cleaned[0][0])

    for i in range(1, len(cleaned)):
        prev_box = cleaned[i - 1][1]
        curr_text, curr_box = cleaned[i]
        gap = curr_box[1] - (prev_box[1] + prev_box[3])

        if gap > para_gap:
            out_lines.append("")  # blank line between paragraphs
        out_lines.append(curr_text)

    return "\n".join(out_lines).strip() + "\n"


# -----------------------------
# OCR model wrapper
# -----------------------------
class HandwritingOCR:
    def __init__(self, model_name: str = "microsoft/trocr-small-handwritten", max_length: int = 256):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.max_length = max_length

        # Optional GPU boost if available
        self.device = "cuda" if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available() else "cpu"
        self.model.to(self.device)

    def ocr_line(self, line_img_gray_or_bgr: np.ndarray) -> str:
        if len(line_img_gray_or_bgr.shape) == 2:
            rgb = cv2.cvtColor(line_img_gray_or_bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(line_img_gray_or_bgr, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb)
        pixel_values = self.processor(images=pil, return_tensors="pt").pixel_values.to(self.device)

        ids = self.model.generate(pixel_values, max_length=self.max_length)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.strip()


# -----------------------------
# Main per-image pipeline
# -----------------------------
def ocr_page_from_bgr(
    img_bgr: np.ndarray,
    filename: str,
    ocr: HandwritingOCR,
    debug: bool = True,
    downscale_max_width: int = 2000,
) -> Tuple[OCRPage, Optional[np.ndarray]]:
    """
    Returns OCRPage plus optional debug visualization image.
    """
    warped = find_page_and_warp(img_bgr)

    # Downscale for speed (big win for phone photos)
    h, w = warped.shape[:2]
    if w > downscale_max_width:
        scale = downscale_max_width / w
        warped = cv2.resize(warped, (downscale_max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

    inv = preprocess_for_segmentation(warped)
    boxes = segment_lines(inv)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    texts: List[str] = []
    lines: List[OCRLine] = []

    for box in boxes:
        crop = crop_with_padding(gray, box, pad=10)
        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
        crop_th = cv2.adaptiveThreshold(
            crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        text = ocr.ocr_line(crop_th)
        texts.append(text)
        lines.append(OCRLine(bbox=box, text=text))

    full_text = stitch_lines_with_paragraphs(texts, boxes)

    dbg = None
    if debug:
        dbg = warped.copy()
        for (x, y, bw, bh) in boxes:
            cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (0, 255, 255), 2)

    return OCRPage(filename=filename, text=full_text, lines=lines), dbg


def make_zip_bytes(pages: List[OCRPage], debug_images: Dict[str, np.ndarray]) -> bytes:
    """
    Build an in-memory ZIP containing txt/json outputs and optional debug line-box images.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        index = []
        for page in pages:
            base = os.path.splitext(page.filename)[0]
            txt_name = f"{base}.txt"
            json_name = f"{base}.json"

            z.writestr(txt_name, page.text)

            payload = {
                "file": page.filename,
                "text": page.text,
                "lines": [{"bbox": list(l.bbox), "text": l.text} for l in page.lines],
            }
            z.writestr(json_name, json.dumps(payload, indent=2, ensure_ascii=False))

            index.append({"file": page.filename, "txt": txt_name, "json": json_name})

            # debug image
            if base in debug_images:
                dbg_bgr = debug_images[base]
                ok, jpg = cv2.imencode(".jpg", dbg_bgr)
                if ok:
                    z.writestr(f"_debug/{base}_lines.jpg", jpg.tobytes())

        z.writestr("results_index.json", json.dumps(index, indent=2, ensure_ascii=False))

    return buf.getvalue()
