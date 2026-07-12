"""
Report generation module.
Uses LLM-powered natural language date parsing to filter invoices
by period and export matching records to Excel.
"""

import json
import re
from datetime import datetime

import pandas as pd

from ocr_engine import get_llm
from indexing import get_all_invoices


# ------------------------------
# LLM-Powered Date Resolution
# ------------------------------
def _resolve_dates_with_llm(period_text: str):
	"""
	Use the Groq LLM to interpret any natural language period description
	and return a (start_date, end_date) tuple of date objects.

	Handles inputs like: "August 2025", "last 3 months", "Q1 2025",
	"before March", "financial year 2024-25", "past 90 days", etc.
	"""
	today = datetime.now().strftime("%Y-%m-%d")

	prompt = (
		"You are a date interpretation assistant. Today's date is " + today + ".\n"
		"The user wants to filter invoices for a specific time period.\n"
		"Interpret the following period description and return the exact start and end dates.\n\n"
		"Rules:\n"
		"- Return ONLY valid JSON: {\"start_date\": \"YYYY-MM-DD\", \"end_date\": \"YYYY-MM-DD\"}\n"
		"- For a single month like 'August 2025', return the first and last day of that month.\n"
		"- For relative terms like 'last month', 'past 90 days', 'this week', calculate from today.\n"
		"- For 'before X', use '1970-01-01' as start_date.\n"
		"- For 'after X', use '2099-12-31' as end_date.\n"
		"- For quarters like 'Q1 2025', return the first and last day of that quarter.\n"
		"- Do NOT include any explanation, markdown, or extra text. ONLY the JSON object.\n\n"
		f"Period: \"{period_text}\"\n"
	)

	try:
		llm = get_llm()
		response = llm.invoke(prompt)
		content = getattr(response, "content", str(response)).strip()

		# Strip markdown code fences if present
		if content.startswith("```"):
			match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
			if match:
				content = match.group(1).strip()

		result = json.loads(content)
		start_date = datetime.strptime(result["start_date"], "%Y-%m-%d").date()
		end_date = datetime.strptime(result["end_date"], "%Y-%m-%d").date()
		return start_date, end_date
	except Exception:
		return None, None


# ------------------------------
# Invoice Filtering
# ------------------------------
def _build_invoice_references(invoices: list[dict]):
	"""Build a lightweight reference list and ID lookup from invoice records."""
	references = []
	id_lookup = {}
	for idx, inv in enumerate(invoices):
		lookup_id = str(inv.get("invoice_number") or f"row-{idx+1}")
		ref = {
			"invoice_id": lookup_id,
			"invoice_number": inv.get("invoice_number"),
			"invoice_date": inv.get("invoice_date"),
			"total_amount": inv.get("total_amount"),
		}
		references.append(ref)
		id_lookup[lookup_id] = idx
	return references, id_lookup


def _parse_invoice_date(value):
	"""Parse an invoice date string into a date object."""
	if not value:
		return None
	try:
		return datetime.strptime(str(value), "%Y-%m-%d").date()
	except Exception:
		try:
			import dateparser
			parsed = dateparser.parse(str(value))
			return parsed.date() if parsed else None
		except Exception:
			return None


def _select_invoices_for_period(period_text: str, invoice_refs: list[dict]):
	"""
	Use the LLM to resolve a natural language period into date boundaries,
	then filter invoice references that fall within that range.
	"""
	if not invoice_refs:
		return [], "No invoice metadata available."

	start_date, end_date = _resolve_dates_with_llm(period_text)
	if not start_date:
		return [], f"Could not interpret time period '{period_text}'. Try formats like 'August 2025', 'last 3 months', 'Q1 2025', etc."

	selected_ids = []
	for ref in invoice_refs:
		inv_date = _parse_invoice_date(ref.get("invoice_date", ""))
		if inv_date and start_date <= inv_date <= end_date:
			selected_ids.append(ref["invoice_id"])

	summary = f"Matched {len(selected_ids)} invoice(s) for period '{period_text}' (dates from {start_date} to {end_date})"
	return selected_ids, summary


# ------------------------------
# Excel Report Generation
# ------------------------------
def create_invoice_excel_report(index_path: str, period_text: str, output_path: str) -> str:
	"""Filter invoices by period and export matching records to an Excel spreadsheet."""
	all_invoice_data = get_all_invoices(index_path)
	if not all_invoice_data:
		return "No invoices found. Please process invoices first."

	invoice_refs, id_lookup = _build_invoice_references(all_invoice_data)
	selected_ids, summary = _select_invoices_for_period(period_text, invoice_refs)

	if not selected_ids:
		error_msg = f"Could not match any invoices to '{period_text}'."
		if summary:
			error_msg += f"\n\n{summary}"
		return error_msg

	filtered = [all_invoice_data[id_lookup[iid]] for iid in selected_ids if iid in id_lookup]

	if not filtered:
		return f"No invoices matched the period '{period_text}'."

	try:
		excel_data = []
		for inv in filtered:
			row = inv.copy()
			row.pop("id", None)
			row.pop("created_at", None)
			items = row.get("items", [])
			if isinstance(items, list):
				row["items"] = "; ".join(items)
			excel_data.append(row)

		pd.DataFrame(excel_data).to_excel(output_path, index=False, engine="openpyxl")
		reason = f" Reason: {summary}" if summary else ""
		return f"Excel report with {len(filtered)} invoices created for '{period_text}' at '{output_path}'.{reason}"
	except Exception as e:
		return f"Failed to create the Excel report: {e}"
