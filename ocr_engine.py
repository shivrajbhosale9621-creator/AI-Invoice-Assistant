"""
OCR engine with OpenCV preprocessing and Groq Vision LLM integration.
Handles the full pipeline: PDF -> image rendering -> CV preprocessing ->
Tesseract OCR -> Vision LLM extraction.
"""

import os
import io
import re
import base64

import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
from PIL import Image

from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables import ConfigurableField
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL_NAME, GROQ_VISION_MODEL_NAME
from models import InvoiceDetails


# ------------------------------
# LLM Factory
# ------------------------------
def get_llm(model_name=None):
	"""Create a ChatGroq LLM instance with the specified or default model."""
	if not GROQ_API_KEY:
		raise ValueError("Missing GROQ API key. Set GROQ_API_KEY in your environment or Streamlit secrets.")
	return ChatGroq(
		groq_api_key=GROQ_API_KEY,
		model_name=model_name or GROQ_MODEL_NAME,
		temperature=0
	).configurable_fields(
		callbacks=ConfigurableField(
			id='callbacks',
			name='callbacks',
			description='A list of callbacks to use for streaming'
		)
	)


# ------------------------------
# Computer Vision Preprocessing
# ------------------------------
def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
	"""
	Apply OpenCV preprocessing to improve OCR accuracy:
	1. Convert to grayscale
	2. Resize 2x for better character recognition
	3. Binarize with Otsu's thresholding
	"""
	open_cv_image = np.array(pil_img)
	if len(open_cv_image.shape) == 3:
		open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
	else:
		return pil_img

	gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	binarized = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	return Image.fromarray(binarized)


# ------------------------------
# Hybrid Vision + OCR Extraction
# ------------------------------
async def extract_invoice_details_from_pdf(file_path: str, llm_vision: ChatGroq) -> InvoiceDetails:
	"""
	Extract structured invoice details from a PDF using a hybrid pipeline:
	1. Render each PDF page to an image via PyMuPDF
	2. Preprocess the image with OpenCV for OCR
	3. Run Tesseract OCR on the preprocessed image
	4. Send both the OCR text and the base64-encoded image to a Groq Vision LLM
	5. Parse the structured JSON response into an InvoiceDetails Pydantic model
	"""
	parser = PydanticOutputParser(pydantic_object=InvoiceDetails)
	format_instructions = parser.get_format_instructions()

	try:
		doc = fitz.open(file_path)
	except Exception as e:
		raise ValueError(f"Failed to open PDF file: {e}")

	pages_data = []
	for page_num in range(len(doc)):
		page = doc[page_num]
		try:
			pix = page.get_pixmap(dpi=150)
			img_data = pix.tobytes("png")
			image = Image.open(io.BytesIO(img_data))

			processed_pil = preprocess_image_for_ocr(image)
			ocr_text = pytesseract.image_to_string(processed_pil)

			buffered = io.BytesIO()
			processed_pil.save(buffered, format="PNG")
			img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

			pages_data.append({"ocr_text": ocr_text, "base64_image": img_str})
		except Exception as e:
			page_text = page.get_text()
			pages_data.append({"ocr_text": page_text, "base64_image": None})

	combined_ocr_text = "\n\n--- PAGE BREAK ---\n\n".join([p["ocr_text"] for p in pages_data])

	prompt_text = (
		"You are a professional invoice extraction system.\n"
		"Analyze the provided invoice document. You have both the raw OCR text and the binarized visual image(s) of the pages.\n"
		"Cross-reference the layout and text to extract the correct values. Be extremely precise with numbers, dates, status, and currencies.\n"
		"If a field is not present in the document, return an empty string for text fields, or 0.0 for numeric fields.\n\n"
		f"{format_instructions}\n\n"
		f"Raw OCR Text:\n{combined_ocr_text}"
	)

	prompt_content = [{"type": "text", "text": prompt_text}]

	image_count = 0
	for p in pages_data:
		if p["base64_image"] is not None:
			prompt_content.append({
				"type": "image_url",
				"image_url": {"url": f"data:image/png;base64,{p['base64_image']}"}
			})
			image_count += 1
			if image_count >= 3:
				break

	message = HumanMessage(content=prompt_content)
	response = await llm_vision.ainvoke([message])

	content = getattr(response, "content", str(response)).strip()
	if content.startswith("```"):
		match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
		if match:
			content = match.group(1).strip()

	result: InvoiceDetails = parser.parse(content)
	return result
