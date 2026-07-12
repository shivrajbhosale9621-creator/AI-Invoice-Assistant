# AI Invoice Automation System

An enterprise-grade document processing and conversational analytics system that automates the extraction, indexing, querying, and reporting of PDF invoices. The application integrates computer vision preprocessing, OCR extraction, multimodal AI reasoning, and Retrieval-Augmented Generation (RAG) to provide a unified financial dashboard.

---

## Architecture Overview

The system is built with a modular, decoupled Python architecture, separating presentation from core services to ensure maintainability, testability, and clean code principles.

- **config.py**: Handles environment variables, secrets management, and default model paths.
- **models.py**: Defines the strict schema for invoice details using Pydantic validation.
- **ocr_engine.py**: Encapsulates computer vision preprocessing operations, PyTesseract OCR, and ChatGroq initialization.
- **indexing.py**: Integrates LlamaIndex document store operations, self-healing data migrations, and conversation history-aware querying.
- **report_generator.py**: Orchestrates LLM-powered natural language date parsing, invoice filtering, and Excel spreadsheet exports.
- **app.py**: Provides a clean Streamlit interface with a centralized corporate theme.

---

## Technical Highlights

### 1. Hybrid Vision and OCR Pipeline
Rather than relying on plain PDF text parsing (which fails on scanned images or complex layouts), this system combines computer vision with multimodal LLMs:
- **Computer Vision Preprocessing (OpenCV)**: PDF pages are rendered as images, upscaled 2x to clarify small fonts, converted to grayscale, and binarized using Otsu's thresholding to maximize text-to-background contrast.
- **Layout-Aware OCR (Tesseract)**: Raw layout-aware text coordinates are extracted from the binarized images.
- **Multimodal AI Reasoning (Groq Vision)**: The high-contrast image and the raw OCR text are fed together into a multimodal LLM (Llama 4 Scout), allowing the model to cross-reference layout visual cues with semantic text blocks to output structured JSON matching the Pydantic schema.

### 2. Hybrid Retrieval for RAG
Semantic vector search is notoriously weak at retrieving highly structured data containing identical keys. To solve this, the Q&A retrieval engine implements a hybrid strategy:
- **Vector Embeddings (LlamaIndex)**: Fetches documents using semantic similarity search.
- **Exact Keyword Fallback**: Extracts query tokens (excluding stop words) and performs a direct substring match against document fields. Keyword matches are merged with semantic results, ensuring queries for specific entities (like names or invoice numbers) are always found.
- **Global Context Injection**: Dynamically appends a lightweight Markdown summary table of all processed invoices to the prompt. This allows the model to answer macro questions (e.g., "what is the total spent?", "how many invoices are unpaid?") that standard localized RAG retrievers cannot calculate.

### 3. Self-Healing Vendor Name Normalization
To prevent duplicate vendor listings in metrics and dashboard charts due to spelling or casing inconsistencies (e.g., "Superstore" vs "SuperStore"), the ingestion pipeline runs an automated normalization step:
- Maps name tokens to clean, lowercase forms.
- On index loading, a background migration automatically standardizes all past documents to the most frequent variant spelling.
- Automatically normalizes all new invoice extractions against the established index.

---

## Features

- **Executive Analytics Dashboard**: Visualize total spend, average invoice value, document count, and unique vendors with responsive charts (Spend by Vendor, Spend Over Time, Payment Status, and Paid vs. Outstanding).
- **Interactive Q&A Chat**: Query invoices using natural language with full multi-turn conversation memory.
- **Visual Invoice Management**: View complete extracted fields, financial summaries, line items, and delete records selectively from the index.
- **Natural Language Reporting**: Export custom reports to Excel by typing dates in natural language (e.g., "August 2025", "last 3 months", "Q1 2025", "before March 2025").

---

## Installation and Setup

### Prerequisites
- Python 3.9 or higher
- Tesseract OCR installed on the system path

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI_Invoice_Automation.git
   cd AI_Invoice_Automation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your API key:
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Launch the application:
   ```bash
   streamlit run app.py
   ```
