"""
Centralized configuration for the Invoice Automation application.
Loads API keys, model names, and default paths from environment variables,
.env files, and Streamlit secrets.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Default Paths ---
DEFAULT_INDEX_PATH = "./invoice_index"
DEFAULT_INBOX_DIR = "./invoices_to_process"

# --- API Keys & Model Names ---
DEFAULT_GROQ_API_KEY = "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"

_SECRETS_LOCATIONS = [
	Path.home() / ".streamlit" / "secrets.toml",
	Path.cwd() / ".streamlit" / "secrets.toml",
]
HAS_STREAMLIT_SECRETS = any(path.exists() for path in _SECRETS_LOCATIONS)


def _get_secret(key: str, default: str = "") -> str:
	"""Attempt to read a key from Streamlit secrets.toml, falling back to a default."""
	if not HAS_STREAMLIT_SECRETS:
		return default
	try:
		import streamlit as st
		return st.secrets[key]
	except (KeyError, Exception):
		return default


GROQ_API_KEY = os.getenv("GROQ_API_KEY") or _get_secret("GROQ_API_KEY", DEFAULT_GROQ_API_KEY)
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME") or _get_secret("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
GROQ_VISION_MODEL_NAME = os.getenv("GROQ_VISION_MODEL_NAME") or _get_secret("GROQ_VISION_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
