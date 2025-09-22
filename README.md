# 📄 AI-Powered Invoice Assistant

> An intelligent document analysis tool that uses a Large Language Model (LLM) to extract, summarize, and answer questions from PDF invoices.

This application provides a user-friendly interface built with Streamlit to manage and query invoice data. It moves beyond simple keyword searching by leveraging an LLM to understand context and provide conversational, human-like answers.


## ✨ Key Features

* **PDF Ingestion:** Upload multiple PDF invoices to create a searchable, indexed knowledge base using LlamaIndex.
* **Interactive Data Viewer:** A dedicated tab to view all indexed invoices, including parsed data (customer, total, items) and the raw extracted text for debugging.
* **Automated Excel Reporting:** Generate and download Excel summaries of your invoices based on natural language time periods (e.g., "August 2025", "last month").
* **LLM-Powered Q&A:** Ask complex questions in plain English and receive intelligent, conversational answers. The system is **industry-agnostic** and works with any type of invoice (e.g., electronics, automotive, agriculture).

## 🧠 How the AI Works

The core of this application is a sophisticated, two-stage process that uses a Large Language Model for both understanding and responding:

1.  **Intelligent Query Generation:** When you ask a question like, "How many printers did we buy from Charlie Davis?", the system doesn't just search for "printer". Instead, it sends your question to an LLM, which generates multiple, high-quality search queries (e.g., "printer quantity Charlie Davis", "printers purchased from Davis", "invoice items printer").
2.  **Conversational Response Synthesis:** The system uses these queries to retrieve the most relevant information from the indexed invoices. It then sends this raw data, along with your original question, back to the LLM. The LLM's final task is to analyze the data and craft a clear, concise, and human-like answer.

This Retrieval-Augmented Generation (RAG) approach makes the assistant flexible, accurate, and capable of handling questions it has never seen before.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Data Processing:** Pandas, DateParser
* **AI & RAG Framework:** LlamaIndex
* **LLM Integration:** LangChain, Groq API (for Llama 3.1)
* **Embeddings:** HuggingFace Sentence-Transformers (local, no API key needed)
* **Environment Management:** `python-dotenv`

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.9+
* A free Groq API Key
## 🏛️ Project Architecture

The application is built with a clean separation of concerns:

* **`app.py` (Frontend):** Manages the Streamlit user interface, handles user inputs, and displays the results. It calls the backend functions to perform the actual work.
* **`invoice_tools.py` (Backend):** Contains all the core logic, including PDF ingestion, LlamaIndex management, data parsing, and the LLM-powered Q&A engine. This file handles all the heavy lifting.

This modular design makes the project easy to understand, maintain, and extend.