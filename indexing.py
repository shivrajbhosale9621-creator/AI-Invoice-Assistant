"""
LlamaIndex indexing and querying operations.
Handles embedding model initialization, index creation/loading,
document retrieval, and conversational Q&A over invoices.
"""

import os
import json
import re

from llama_index.core import (
	Document,
	StorageContext,
	VectorStoreIndex,
	load_index_from_storage,
	Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_core.messages import HumanMessage, AIMessage

from config import DEFAULT_INDEX_PATH, GROQ_VISION_MODEL_NAME
from ocr_engine import get_llm, extract_invoice_details_from_pdf


# ------------------------------
# Embedding Model (Lazy Init)
# ------------------------------
_embed_model = None
_embed_model_initialized = False


def _get_embed_model():
	"""Lazily initialize the HuggingFace embedding model for LlamaIndex."""
	global _embed_model, _embed_model_initialized
	if not _embed_model_initialized:
		_embed_model_initialized = True
		try:
			os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
			_embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
			Settings.embed_model = _embed_model
		except NotImplementedError as e:
			if "meta tensor" in str(e).lower():
				try:
					os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
					_embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
					Settings.embed_model = _embed_model
				except Exception:
					Settings.embed_model = None
			else:
				raise
		except Exception:
			Settings.embed_model = None
	return _embed_model


# ------------------------------
# Index Operations
# ------------------------------
def load_index(index_path: str, normalize: bool = True):
	"""Load a persisted LlamaIndex vector store index from disk."""
	_get_embed_model()
	storage_context = StorageContext.from_defaults(persist_dir=index_path)
	index = load_index_from_storage(storage_context)
	if normalize:
		_normalize_all_vendors_in_index(index_path)
		# Reload index with updated docstore text contents
		storage_context = StorageContext.from_defaults(persist_dir=index_path)
		index = load_index_from_storage(storage_context)
	return index


def normalize_vendor_name(name: str, existing_names: set) -> str:
	"""Standardize vendor name against existing ones using case-insensitive and punctuation-free checks."""
	if not name:
		return "Unknown"
	name_stripped = name.strip()
	name_lower = name_stripped.lower()
	
	# 1. Exact case-insensitive match
	for existing in existing_names:
		if existing.lower() == name_lower:
			return existing
			
	# 2. Match after removing non-alphanumeric noise
	clean = lambda s: re.sub(r'[^a-zA-Z0-9]', '', s).lower()
	clean_name = clean(name_stripped)
	for existing in existing_names:
		if clean(existing) == clean_name:
			return existing
			
	existing_names.add(name_stripped)
	return name_stripped


def _normalize_all_vendors_in_index(index_path: str):
	"""Read index, group similar vendor name keys, and overwrite variants with the most common casing format."""
	try:
		index = load_index(index_path, normalize=False)
		changed = False
		
		# Count casing and spacing frequencies of all unique vendor names in store
		vendor_counts = {}
		for doc in index.docstore.docs.values():
			payload = _extract_json_dict(doc.get_content())
			vname = payload.get("vendor_name")
			if vname:
				vname_stripped = vname.strip()
				vendor_counts[vname_stripped] = vendor_counts.get(vname_stripped, 0) + 1
				
		# Build a canonical map mapping simplified names to their most popular spelling
		canonical_map = {}
		clean = lambda s: re.sub(r'[^a-zA-Z0-9]', '', s).lower()
		for name, count in sorted(vendor_counts.items(), key=lambda x: x[1], reverse=True):
			cl = clean(name)
			if cl not in canonical_map:
				canonical_map[cl] = name
				
		# Update node values directly inside the document store
		for doc_id, doc in index.docstore.docs.items():
			payload = _extract_json_dict(doc.get_content())
			vname = payload.get("vendor_name")
			if vname:
				cl = clean(vname.strip())
				canonical = canonical_map.get(cl)
				if canonical and vname != canonical:
					payload["vendor_name"] = canonical
					doc.text = json.dumps(payload, indent=2)
					changed = True
					
		if changed:
			index.storage_context.persist(persist_dir=index_path)
	except Exception:
		pass


def _extract_json_dict(text: str) -> dict:
	"""Attempt to parse a JSON object from a string, with fallback regex extraction."""
	if not text:
		return {}
	text = text.strip()
	try:
		return json.loads(text)
	except Exception:
		pass
	match = re.search(r"\{.*\}", text, re.DOTALL)
	if match:
		try:
			return json.loads(match.group(0))
		except Exception:
			return {}
	return {}


def get_all_invoices(index_path=DEFAULT_INDEX_PATH) -> list[dict]:
	"""Retrieve all invoice records from the LlamaIndex docstore."""
	try:
		index = load_index(index_path)
		invoices = []
		for doc in index.docstore.docs.values():
			payload = _extract_json_dict(doc.get_content())
			if payload and ("invoice_number" in payload or "vendor_name" in payload):
				invoices.append(payload)
		return invoices
	except Exception as e:
		import streamlit as st
		st.warning(f"Could not load invoice index: {e}")
		return []


def get_all_invoices_with_ids(index_path=DEFAULT_INDEX_PATH) -> list[dict]:
	"""Retrieve unique invoice records with their docstore node IDs for management operations."""
	try:
		index = load_index(index_path)
		results = []
		seen_keys = set()
		for doc_id, doc in index.docstore.docs.items():
			payload = _extract_json_dict(doc.get_content())
			if payload and ("invoice_number" in payload or "vendor_name" in payload):
				key = (payload.get("invoice_number"), payload.get("vendor_name"))
				if key in seen_keys:
					continue
				seen_keys.add(key)
				payload["_doc_id"] = doc_id
				results.append(payload)
		return results
	except Exception:
		return []


def delete_invoice(index_path: str, doc_id: str) -> bool:
	"""Delete a single invoice from the LlamaIndex index by its docstore node ID or ref_doc_id."""
	try:
		index = load_index(index_path)
		node = index.docstore.docs.get(doc_id)
		ref_doc_id = getattr(node, "ref_doc_id", doc_id) or doc_id

		try:
			index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)
		except Exception:
			try:
				index.delete_ref_doc(doc_id, delete_from_docstore=True)
			except Exception:
				pass

		if hasattr(index, "vector_store") and hasattr(index.vector_store, "delete"):
			try:
				index.vector_store.delete(doc_id)
			except Exception:
				pass
			try:
				index.vector_store.delete(ref_doc_id)
			except Exception:
				pass

		if doc_id in index.docstore.docs:
			del index.docstore.docs[doc_id]
		if ref_doc_id in index.docstore.docs:
			del index.docstore.docs[ref_doc_id]

		index.storage_context.persist(persist_dir=index_path)
		return True
	except Exception as e:
		import streamlit as st
		st.error(f"Failed to delete: {e}")
		return False


async def load_invoices_to_index(directory: str, index_path: str, llm) -> str:
	"""
	Process all PDF invoices in a directory:
	1. Extract details using the hybrid Vision+OCR pipeline
	2. Create LlamaIndex Document nodes from the structured JSON
	3. Build and persist a VectorStoreIndex
	"""
	try:
		invoice_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
		if not invoice_files:
			return f"No PDF files found in '{directory}'."
	except FileNotFoundError:
		return f"Error: The directory '{directory}' does not exist."

	# Gather existing vendor names to normalize against
	existing_vendors = set()
	try:
		existing_invoices = get_all_invoices(index_path)
		for item in existing_invoices:
			name = item.get("vendor_name")
			if name:
				existing_vendors.add(name.strip())
	except Exception:
		pass

	storage_context = StorageContext.from_defaults()
	documents_to_index = []
	processed = 0

	llm_vision = get_llm(GROQ_VISION_MODEL_NAME)

	for name in invoice_files:
		file_path = os.path.join(directory, name)
		try:
			inv = await extract_invoice_details_from_pdf(file_path, llm_vision)
			
			# Normalize the extracted vendor name
			normalized_vendor = normalize_vendor_name(inv.vendor_name, existing_vendors)
			inv.vendor_name = normalized_vendor
			
			doc = Document(text=json.dumps(inv.model_dump(), indent=2), metadata={'file_name': name})
			documents_to_index.append(doc)
			processed += 1
		except Exception as e:
			msg = str(e)
			if "invalid api key" in msg.lower():
				return "Error: Invalid GROQ API Key. Update the GROQ_API_KEY environment variable and retry."
			import streamlit as st
			st.warning(f"Failed to process {name}: {msg}")

	if documents_to_index:
		_get_embed_model()
		index = VectorStoreIndex.from_documents(documents_to_index, storage_context=storage_context)
		os.makedirs(index_path, exist_ok=True)
		index.storage_context.persist(persist_dir=index_path)
	return f"Processed and indexed {processed} of {len(invoice_files)} invoices from '{directory}'."


def query_invoice_data(index_path: str, question: str, chat_history: list = None) -> str:
	"""
	Query invoice data with conversation memory using RAG.
	Retrieves relevant invoice nodes, builds context with chat history,
	and generates an answer via Groq LLM.
	"""
	if chat_history is None:
		chat_history = []

	try:
		index = load_index(index_path)
	except Exception:
		return "Error: The invoice index does not exist. Please process invoices first."
	retriever = index.as_retriever(similarity_top_k=5)
	nodes = retriever.retrieve(question)
	
	# Hybrid search: Scan docstore for keywords in query to ensure exact matches are found
	stop_words = {"what", "items", "we", "have", "brought", "from", "the", "a", "of", "and", "in", "for", "to", "is", "are", "on", "with", "about", "query", "question", "invoices", "invoice", "show", "me", "find", "get"}
	words = re.findall(r'\b\w{3,}\b', question.lower())
	keywords = [w for w in words if w not in stop_words]
	
	keyword_nodes = []
	seen_contents = {node.get_content() for node in nodes}
	
	if keywords:
		for doc in index.docstore.docs.values():
			content = doc.get_content()
			if content in seen_contents:
				continue
			content_lower = content.lower()
			if any(kw in content_lower for kw in keywords):
				keyword_nodes.append(doc)
				seen_contents.add(content)
	
	all_nodes = keyword_nodes + nodes
	all_nodes = all_nodes[:10]  # Limit to top 10 nodes

	# Build a summary table of all invoices to support global aggregation queries
	all_invoices = get_all_invoices(index_path)
	summary_lines = ["| Invoice # | Vendor | Date | Total | Status | Currency |", "|---|---|---|---|---|---|"]
	for inv in all_invoices:
		num = inv.get("invoice_number") or "N/A"
		vendor = inv.get("vendor_name") or "Unknown"
		date = inv.get("invoice_date") or "N/A"
		total = inv.get("total_amount") or 0.0
		status = inv.get("payment_status") or "Unpaid"
		curr = inv.get("currency") or "USD"
		summary_lines.append(f"| {num} | {vendor} | {date} | {total} | {status} | {curr} |")
	
	global_summary = "\n".join(summary_lines)

	if not all_nodes:
		context = "No specific detailed invoice records retrieved."
	else:
		context_parts = []
		for node in all_nodes:
			meta = node.metadata or {}
			label = meta.get("file_name", "Invoice document")
			context_parts.append(f"Source: {label}\n{node.get_content()}")
		context = "\n\n---\n\n".join(context_parts)

	history_context = ""
	if chat_history:
		history_context = "\n\nPrevious conversation:\n"
		for msg in chat_history[-6:]:
			if isinstance(msg, HumanMessage):
				history_context += f"User: {msg.content}\n"
			elif isinstance(msg, AIMessage):
				history_context += f"Assistant: {msg.content}\n"

	prompt = (
		"You are an assistant that analyzes structured invoice data in JSON format.\n"
		"Answer the user's question using the provided invoice details and global summary table.\n"
		"If the user's question requires aggregating, summing, counting, listing, or summarizing all invoices (e.g. 'what is the total spent?', 'list all vendors', 'how many invoices do we have?'), use the Global Summary Table to calculate the exact answer.\n"
		"If the question is about a specific invoice or itemized details, refer to the detailed Invoice Entries.\n"
		"If the answer is not present or cannot be calculated from the provided data, say you do not know.\n\n"
		f"### Global Summary Table (All Invoices):\n{global_summary}\n\n"
		f"### Detailed Invoice Entries (Selected):\n{context}\n\n"
		f"{history_context}\n"
		f"Current question: {question}\nAnswer:"
	)
	llm = get_llm()
	response = llm.invoke(prompt)
	return getattr(response, "content", str(response))
