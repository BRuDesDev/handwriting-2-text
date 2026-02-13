import io
import os
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from ocr_engine import HandwritingOCR, ocr_page_from_bgr, make_zip_bytes

app = FastAPI(title="Handwriting OCR Web App")
templates = Jinja2Templates(directory="templates")

OCR_MODEL: Optional[HandwritingOCR] = None
MAX_FILES_DEFAULT = 100

@app.on_event("startup")
def load_model_once():
    global OCR_MODEL
    # Loads once when the server starts (still takes time first run, but cleaner)
    OCR_MODEL = HandwritingOCR(model_name="microsoft/trocr-small-handwritten")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "max_files": MAX_FILES_DEFAULT})


@app.post("/ocr")
async def ocr_endpoint(
    files: List[UploadFile] = File(...),
    debug: bool = Form(True),
    max_files: int = Form(MAX_FILES_DEFAULT),
):
    global OCR_MODEL
    if OCR_MODEL is None:
        # Safety net
        OCR_MODEL = HandwritingOCR(model_name="microsoft/trocr-small-handwritten")

    if len(files) > max_files:
        return {"error": f"Too many files: {len(files)}. Max allowed is {max_files}."}

    pages = []
    debug_images = {}

    for f in files:
        data = await f.read()
        npbuf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
        if img is None:
            continue

        page, dbg = ocr_page_from_bgr(
            img_bgr=img,
            filename=f.filename,
            ocr=OCR_MODEL,
            debug=debug,
            downscale_max_width=2000,
        )
        pages.append(page)

        if debug and dbg is not None:
            base = os.path.splitext(f.filename)[0]
            debug_images[base] = dbg

    zip_bytes = make_zip_bytes(pages, debug_images)

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=handwriting_ocr_results.zip"},
    )
