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

        # Extract text using pdfplumber
        pages_data = []
        full_text_parts = []

        with pdfplumber.open(file.stream) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF has {total_pages} pages")

            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text with layout preservation for better spacing
                # pdfplumber does a better job than Go libraries at word spacing
                page_text = page.extract_text(
                    layout=True,
                    x_tolerance=3,  # Horizontal tolerance for detecting word boundaries
                    y_tolerance=3   # Vertical tolerance for detecting line boundaries
                )

                if page_text:
                    # Normalize whitespace: collapse multiple spaces to single space
                    # but preserve paragraph breaks (double newlines)
                    page_text = re.sub(r' +', ' ', page_text)  # Multiple spaces -> single space
                    page_text = re.sub(r'\n +', '\n', page_text)  # Remove leading spaces on lines
                    page_text = re.sub(r' +\n', '\n', page_text)  # Remove trailing spaces on lines
                    page_text = page_text.strip()

                    pages_data.append({
                        "page": page_num,
                        "text": page_text
                    })

                    # Add page marker for full text
                    full_text_parts.append(f"--- Page {page_num} ---\n{page_text}")
                else:
                    logger.warning(f"No text extracted from page {page_num}")
                    pages_data.append({
                        "page": page_num,
                        "text": ""
                    })

            # Get PDF metadata
            metadata = pdf.metadata or {}

        # Combine all pages
        full_text = "\n\n".join(full_text_parts)

        logger.info(f"Successfully extracted {len(full_text)} characters from {total_pages} pages")

        return jsonify({
            "success": True,
            "text": full_text,
            "pages": pages_data,
            "total_pages": total_pages,
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
