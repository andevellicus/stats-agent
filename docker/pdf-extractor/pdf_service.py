#!/usr/bin/env python3
"""
PDF Text Extraction Microservice using pdfplumber

Provides a simple HTTP API for extracting text from PDFs with proper spacing.
"""

import os
import re
import logging
from flask import Flask, request, jsonify
import pdfplumber
from io import BytesIO
import statistics
from pdfminer.high_level import extract_text as pm_extract_text
from pdfminer.layout import LAParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "pdf-extractor"}), 200

def clean_text(s: str) -> str:
    # Remove soft hyphen and dehyphenate across linebreaks
    s = s.replace("\u00ad", "")
    s = re.sub(r'([A-Za-z0-9])-\r?\n([A-Za-z0-9])', r'\1\2', s)
    # Normalize whitespace, keep paragraph breaks
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def words_based_page_text(page, use_text_flow: bool, xt: float, yt: float) -> str:
    # Reconstruct lines from words using bounding boxes
    words = page.extract_words(use_text_flow=use_text_flow, keep_blank_chars=False, x_tolerance=xt, y_tolerance=yt)
    lines = {}
    for w in words or []:
        top = w.get("top"); bottom = w.get("bottom"); x0 = w.get("x0"); text = w.get("text", "")
        if top is None or bottom is None or x0 is None:
            continue
        y_mid = round((top + bottom) / 2, 1)
        lines.setdefault(y_mid, []).append(w)
    out_lines = []
    for _, ws in sorted(lines.items(), key=lambda kv: kv[0]):
        ws_sorted = sorted(ws, key=lambda w: w.get("x0", 0))
        out_lines.append(" ".join(w.get("text", "") for w in ws_sorted))
    return clean_text("\n".join(out_lines))

def pdfminer_page_text(pdf_bytes: bytes, page_index: int, lap: LAParams) -> str:
    try:
        txt = pm_extract_text(BytesIO(pdf_bytes), laparams=lap, page_numbers=[page_index])
    except Exception:
        txt = ""
    return clean_text(txt or "")

def quality_metrics(text: str):
    if not text:
        return (0.0, 999.0)
    spaces = text.count(' ')
    whitespace_ratio = spaces / max(1, len(text))
    words = re.findall(r'\w+', text)
    avg_word_len = statistics.mean(map(len, words)) if words else 999.0
    return (whitespace_ratio, avg_word_len)

def strip_repeated_headers_footers(pages_texts: list[str]) -> list[str]:
    if len(pages_texts) < 3:
        return pages_texts
    def first_nonempty(line_list):
        for l in line_list:
            t = l.strip()
            if t:
                return re.sub(r'\s+', ' ', t)
        return ""
    def last_nonempty(line_list):
        for i in range(len(line_list)-1, -1, -1):
            t = line_list[i].strip()
            if t:
                return re.sub(r'\s+', ' ', t)
        return ""
    tops, bots = [], []
    for txt in pages_texts:
        lines = txt.splitlines()
        tops.append(first_nonempty(lines))
        bots.append(last_nonempty(lines))
    from collections import Counter
    def pick_common(lines):
        if not lines:
            return ""
        common, cnt = Counter(lines).most_common(1)[0]
        return common if common and cnt >= 0.6*len(lines) and 8 <= len(common) <= 200 else ""
    top_c = pick_common(tops)
    bot_c = pick_common(bots)
    if not top_c and not bot_c:
        return pages_texts
    cleaned = []
    for txt in pages_texts:
        lines = txt.splitlines()
        # drop header
        if top_c:
            for i, l in enumerate(lines):
                if l.strip():
                    if re.sub(r'\s+', ' ', l.strip()) == top_c:
                        lines = lines[i+1:]
                    break
        # drop footer
        if bot_c:
            for i in range(len(lines)-1, -1, -1):
                if lines[i].strip():
                    if re.sub(r'\s+', ' ', lines[i].strip()) == bot_c:
                        lines = lines[:i]
                    break
        cleaned.append("\n".join(lines))
    return cleaned

@app.route('/extract', methods=['POST'])
def extract_pdf():
    """
    Extract text from uploaded PDF file

    Request:
        multipart/form-data with 'file' field containing PDF

    Response:
        JSON: {
            "success": true,
            "text": "full extracted text",
            "pages": [
                {"page": 1, "text": "page 1 content"},
                {"page": 2, "text": "page 2 content"}
            ],
            "total_pages": 10,
            "metadata": {...}
        }

    Error Response:
        JSON: {
            "success": false,
            "error": "error message"
        }
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "Empty filename"
            }), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({
                "success": False,
                "error": "File must be a PDF"
            }), 400

        logger.info(f"Processing PDF: {file.filename}")

        # Parse tuning params (with sensible defaults)
        mode = (request.args.get('mode') or '').lower()  # 'words' | 'pdfminer' | '' (auto)
        use_text_flow = (request.args.get('flow') or '').lower() in ('1', 'true', 'yes')
        xt = float(request.args.get('xt') or 3)
        yt = float(request.args.get('yt') or 3)
        wm = float(request.args.get('wm') or 0.0)
        cm = float(request.args.get('cm') or 0.0)
        lm = float(request.args.get('lm') or 0.0)
        bf = float(request.args.get('bf') or 0.0)

        # Read PDF into memory (for using both pdfplumber and pdfminer)
        pdf_bytes = file.read()
        pages_data = []
        full_text_parts = []
        metadata = {}

        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF has {total_pages} pages")

            # Decide strategy in auto mode by sampling the first page
            chosen = mode if mode in ('words', 'pdfminer') else ''
            if not chosen and total_pages > 0:
                try:
                    p0 = pdf.pages[0]
                    cand_w = words_based_page_text(p0, use_text_flow, xt, yt)
                    lap_probe = LAParams(
                        char_margin=cm or 2.0,
                        word_margin=wm or 0.35,
                        line_margin=lm or 0.2,
                        boxes_flow=bf if bf != 0.0 else 0.5,
                        all_texts=True,
                        detect_vertical=True,
                    )
                    cand_p = pdfminer_page_text(pdf_bytes, 0, lap_probe)
                    wr_w, awl_w = quality_metrics(cand_w)
                    wr_p, awl_p = quality_metrics(cand_p)
                    chosen = 'words' if (wr_w, -awl_w) > (wr_p, -awl_p) else 'pdfminer'
                except Exception as e:
                    logger.warning(f"Auto strategy probe failed, defaulting to words: {e}")
                    chosen = 'words'
            if not chosen:
                chosen = 'words'

            if chosen == 'words':
                for i, page in enumerate(pdf.pages, start=1):
                    page_text = words_based_page_text(page, use_text_flow, xt, yt)
                    pages_data.append({"page": i, "text": page_text})
            else:  # pdfminer path
                lap = LAParams(
                    char_margin=cm or 2.0,
                    word_margin=wm or 0.35,
                    line_margin=lm or 0.2,
                    boxes_flow=bf if bf != 0.0 else 0.5,
                    all_texts=True,
                    detect_vertical=True,
                )
                for i in range(total_pages):
                    page_text = pdfminer_page_text(pdf_bytes, i, lap)
                    pages_data.append({"page": i+1, "text": page_text})

            # PDF metadata
            metadata = pdf.metadata or {}

        # Remove repeated headers/footers across pages
        texts = [p["text"] for p in pages_data]
        texts = strip_repeated_headers_footers(texts)
        for i, t in enumerate(texts):
            pages_data[i]["text"] = t

        for p in pages_data:
            full_text_parts.append(f"--- Page {p['page']} ---\n{p['text']}")
        full_text = "\n\n".join(full_text_parts)

        logger.info(f"Successfully extracted {len(full_text)} characters from {len(pages_data)} pages (mode={mode or 'auto'} -> {chosen})")

        return jsonify({
            "success": True,
            "text": full_text,
            "pages": pages_data,
            "total_pages": len(pages_data),
            "metadata": {
                "title": metadata.get('Title', ''),
                "author": metadata.get('Author', ''),
                "subject": metadata.get('Subject', ''),
                "creator": metadata.get('Creator', ''),
            },
            "characters": len(full_text)
        }), 200

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({
        "success": False,
        "error": "File too large. Maximum size is 10MB"
    }), 413

if __name__ == '__main__':
    # For development only - gunicorn is used in production
    app.run(host='0.0.0.0', port=9001, debug=False)
