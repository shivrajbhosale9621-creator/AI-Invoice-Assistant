"""
Invoice Automation App - Streamlit UI
Clean presentation layer with professional styling for invoice processing workflows.
"""

import os
import asyncio

import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

from config import (
	GROQ_API_KEY,
	GROQ_MODEL_NAME,
	GROQ_VISION_MODEL_NAME,
	DEFAULT_INDEX_PATH,
	DEFAULT_INBOX_DIR,
)
from ocr_engine import get_llm
from indexing import load_invoices_to_index, query_invoice_data, get_all_invoices, get_all_invoices_with_ids, delete_invoice
from report_generator import create_invoice_excel_report


# =============================================
# Page Config
# =============================================
st.set_page_config(
	page_title="Invoice Automation",
	page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>I</text></svg>",
	layout="wide",
	initial_sidebar_state="collapsed",
)


# =============================================
# Global CSS - Professional Dark-Nav Theme
# =============================================
st.markdown("""
<style>
	/* ---------- Google Font ---------- */
	@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

	html, body, [class*="css"] {
		font-family: 'Inter', sans-serif;
	}

	/* ---------- Header bar ---------- */
	.app-header {
		background: linear-gradient(135deg, #1B4F72 0%, #154360 100%);
		padding: 1.5rem 2rem;
		border-radius: 12px;
		margin-bottom: 1.5rem;
		color: #fff;
	}
	.app-header h1 {
		margin: 0;
		font-size: 1.75rem;
		font-weight: 700;
		letter-spacing: -0.02em;
	}
	.app-header p {
		margin: 0.25rem 0 0 0;
		font-size: 0.9rem;
		color: rgba(255,255,255,0.75);
	}

	/* ---------- Metric cards ---------- */
	[data-testid="stMetric"] {
		background-color: #ffffff;
		border: 1px solid #E5E8EB;
		padding: 1.25rem 1.5rem;
		border-radius: 10px;
		box-shadow: 0 1px 3px rgba(0,0,0,0.04);
		transition: box-shadow 0.2s ease, border-color 0.2s ease;
	}
	[data-testid="stMetric"]:hover {
		box-shadow: 0 4px 12px rgba(27,79,114,0.10);
		border-color: #1B4F72;
	}
	[data-testid="stMetricLabel"] {
		font-size: 0.8rem !important;
		font-weight: 600 !important;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: #6C757D !important;
	}
	[data-testid="stMetricValue"] {
		font-size: 1.6rem !important;
		font-weight: 700 !important;
		color: #1B4F72 !important;
	}

	/* ---------- Section headers ---------- */
	.section-header {
		font-size: 1rem;
		font-weight: 600;
		color: #343A40;
		margin: 1.5rem 0 0.75rem 0;
		padding-bottom: 0.5rem;
		border-bottom: 2px solid #1B4F72;
		display: inline-block;
	}

	/* ---------- Tab styling ---------- */
	.stTabs [data-baseweb="tab-list"] {
		gap: 0.25rem;
		background-color: #EEF2F5;
		border-radius: 10px;
		padding: 4px;
	}
	.stTabs [data-baseweb="tab"] {
		border-radius: 8px;
		font-weight: 500;
		font-size: 0.875rem;
		padding: 0.5rem 1.25rem;
	}
	.stTabs [aria-selected="true"] {
		background-color: #1B4F72 !important;
		color: #fff !important;
	}

	/* ---------- Sidebar ---------- */
	section[data-testid="stSidebar"] {
		background-color: #1B2A3D;
	}
	section[data-testid="stSidebar"] .stMarkdown,
	section[data-testid="stSidebar"] label,
	section[data-testid="stSidebar"] .stTextInput label,
	section[data-testid="stSidebar"] h2,
	section[data-testid="stSidebar"] h3 {
		color: #D5DBDE !important;
	}
	section[data-testid="stSidebar"] .stTextInput input {
		background-color: #253649;
		border-color: #3A5068;
		color: #E8ECEF;
	}
	section[data-testid="stSidebar"] hr {
		border-color: rgba(255,255,255,0.1);
	}

	/* ---------- Dataframe ---------- */
	.stDataFrame {
		border: 1px solid #E5E8EB;
		border-radius: 8px;
		overflow: hidden;
	}

	/* ---------- Chat messages ---------- */
	[data-testid="stChatMessage"] {
		border-radius: 10px;
		border: 1px solid #E5E8EB;
		margin-bottom: 0.5rem;
	}

	/* ---------- Footer ---------- */
	.app-footer {
		text-align: center;
		padding: 1.5rem 0 0.5rem 0;
		color: #ADB5BD;
		font-size: 0.8rem;
		border-top: 1px solid #E5E8EB;
		margin-top: 3rem;
	}
</style>
""", unsafe_allow_html=True)


# Ensure directories exist
os.makedirs(DEFAULT_INDEX_PATH, exist_ok=True)
os.makedirs(DEFAULT_INBOX_DIR, exist_ok=True)

if not GROQ_API_KEY:
	st.error("Missing GROQ_API_KEY. Please configure the GROQ_API_KEY environment variable to enable document extraction and querying features.")


# =============================================
# Header
# =============================================
st.markdown("""
<div class="app-header">
	<h1>Invoice Automation</h1>
	<p>Upload, extract, query, and export invoice data powered by Vision AI and LlamaIndex</p>
</div>
""", unsafe_allow_html=True)


# =============================================
# Tabs
# =============================================
tab_dash, tab_upload, tab_manage, tab_query, tab_export = st.tabs([
	"Dashboard",
	"Upload & Index",
	"Manage Invoices",
	"Query",
	"Export Report",
])


# --------------------------------------------------
# Tab: Dashboard
# --------------------------------------------------
with tab_dash:
	invoices = get_all_invoices(DEFAULT_INDEX_PATH)

	if not invoices:
		st.info("No invoices processed yet. Go to the **Upload & Index** tab to get started.")
	else:
		df = pd.DataFrame(invoices)

		# -- KPI Metrics --
		total_spend = df["total_amount"].sum()
		avg_spend = df["total_amount"].mean()
		total_count = len(df)
		unique_vendors = df["vendor_name"].nunique()

		col1, col2, col3, col4 = st.columns(4)
		with col1:
			st.metric("Total Spend", f"${total_spend:,.2f}")
		with col2:
			st.metric("Avg Invoice", f"${avg_spend:,.2f}")
		with col3:
			st.metric("Invoices", f"{total_count}")
		with col4:
			st.metric("Vendors", f"{unique_vendors}")

		st.markdown("---")

		# -- Charts Row 1 --
		col_chart1, col_chart2 = st.columns(2)

		with col_chart1:
			st.markdown('<div class="section-header">Spend by Vendor</div>', unsafe_allow_html=True)
			vendor_df = df.groupby("vendor_name")["total_amount"].sum().reset_index()
			vendor_df = vendor_df.sort_values(by="total_amount", ascending=False).set_index("vendor_name")
			st.bar_chart(vendor_df)

		with col_chart2:
			st.markdown('<div class="section-header">Spend Over Time</div>', unsafe_allow_html=True)
			df_dates = df.copy()
			df_dates["parsed_date"] = pd.to_datetime(df_dates["invoice_date"], errors='coerce')
			df_dates = df_dates.dropna(subset=["parsed_date"])

			if not df_dates.empty:
				df_dates["Month-Year"] = df_dates["parsed_date"].dt.to_period("M").astype(str)
				trend_df = df_dates.groupby("Month-Year")["total_amount"].sum().reset_index()
				trend_df = trend_df.sort_values("Month-Year").set_index("Month-Year")
				st.line_chart(trend_df)
			else:
				st.caption("No parseable invoice dates to display trends.")

		st.markdown("---")

		# -- Charts Row 2 --
		col_chart3, col_chart4 = st.columns(2)

		with col_chart3:
			st.markdown('<div class="section-header">Payment Status</div>', unsafe_allow_html=True)
			if "payment_status" in df.columns:
				status_counts = df["payment_status"].value_counts().reset_index()
				status_counts.columns = ["Status", "Count"]
				status_counts = status_counts.set_index("Status")
				st.bar_chart(status_counts)
			else:
				st.caption("Payment status data is not available.")

		with col_chart4:
			st.markdown('<div class="section-header">Paid vs Outstanding</div>', unsafe_allow_html=True)
			if "payment_status" in df.columns and "total_amount" in df.columns:
				df["Spend Type"] = df["payment_status"].apply(
					lambda s: "Paid" if str(s).strip().lower() == "paid" else "Outstanding"
				)
				outstanding_df = df.groupby("Spend Type")["total_amount"].sum().reset_index()
				outstanding_df = outstanding_df.set_index("Spend Type")
				st.bar_chart(outstanding_df)
			else:
				st.caption("Outstanding spend data is not available.")

		st.markdown("---")

		# -- Recent Invoices Table --
		st.markdown('<div class="section-header">Recent Invoices</div>', unsafe_allow_html=True)
		preview_cols = [
			"file_name", "vendor_name", "invoice_number", "invoice_date",
			"due_date", "payment_status", "total_amount", "currency"
		]
		existing_preview_cols = [c for c in preview_cols if c in df.columns]
		st.dataframe(df[existing_preview_cols], use_container_width=True, hide_index=True)


# --------------------------------------------------
# Tab: Upload & Index
# --------------------------------------------------
with tab_upload:
	st.markdown('<div class="section-header">Upload Invoice PDFs</div>', unsafe_allow_html=True)
	st.caption("Select one or more PDF files to upload. They will be saved to the invoices folder and processed using the Vision AI pipeline.")

	uploaded_files = st.file_uploader(
		"Select PDF files",
		type=["pdf"],
		accept_multiple_files=True,
		label_visibility="collapsed",
	)
	if uploaded_files:
		os.makedirs(DEFAULT_INBOX_DIR, exist_ok=True)
		for up in uploaded_files:
			dest = os.path.join(DEFAULT_INBOX_DIR, up.name)
			with open(dest, "wb") as f:
				f.write(up.read())
		st.success(f"Saved {len(uploaded_files)} file(s) to {DEFAULT_INBOX_DIR}")

	st.markdown("---")
	st.markdown('<div class="section-header">Process & Build Index</div>', unsafe_allow_html=True)
	st.caption("This will run the hybrid OCR + Vision extraction pipeline on all PDFs in the invoices folder and build a searchable vector index.")

	col_proc, col_rst = st.columns([1, 1])
	with col_proc:
		if st.button("Process Invoices", type="primary", use_container_width=True):
			if not GROQ_API_KEY:
				st.error("Missing GROQ_API_KEY. Set it in your environment and restart the app.")
			else:
				llm = get_llm()
				with st.spinner("Extracting invoice details and building index..."):
					msg = asyncio.run(load_invoices_to_index(DEFAULT_INBOX_DIR, DEFAULT_INDEX_PATH, llm))
				st.success(msg)
	with col_rst:
		if st.button("Reset Index", type="secondary", use_container_width=True):
			try:
				import shutil
				if os.path.exists(DEFAULT_INDEX_PATH):
					shutil.rmtree(DEFAULT_INDEX_PATH)
				st.success("Index cleared successfully.")
				st.rerun()
			except Exception as e:
				st.warning(f"Could not clear index: {e}")


# --------------------------------------------------
# Tab: Query
# --------------------------------------------------
with tab_query:
	st.markdown('<div class="section-header">Invoice Q&A</div>', unsafe_allow_html=True)
	st.caption("Ask natural language questions about your processed invoices. The system uses RAG to retrieve relevant data and generate answers.")

	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []

	for msg in st.session_state.chat_history:
		if isinstance(msg, HumanMessage):
			with st.chat_message("user"):
				st.write(msg.content)
		elif isinstance(msg, AIMessage):
			with st.chat_message("assistant"):
				st.write(msg.content)

	question = st.chat_input("Type your question here...")

	if question:
		st.session_state.chat_history.append(HumanMessage(content=question))
		with st.spinner("Searching invoice index..."):
			answer = query_invoice_data(DEFAULT_INDEX_PATH, question, st.session_state.chat_history)
		st.session_state.chat_history.append(AIMessage(content=answer))
		st.rerun()

	if st.session_state.chat_history:
		if st.button("Clear conversation", type="secondary"):
			st.session_state.chat_history = []
			st.rerun()


# --------------------------------------------------
# Tab: Export Report
# --------------------------------------------------
with tab_export:
	st.markdown('<div class="section-header">Export to Excel</div>', unsafe_allow_html=True)
	st.caption("Filter invoices by a time period using natural language and export the results to an Excel spreadsheet.")

	with st.expander("Supported date formats", expanded=False):
		st.markdown("""
| Format | Example |
|---|---|
| Month + Year | `January 2025`, `March 2025` |
| Date range | `21/03/2025 to 21/06/2025` |
| Relative | `last month`, `this week`, `past 90 days` |
| Quarter | `Q1 2025`, `Q3 2024` |
| Before / After | `before 15/06/2025`, `after 01/01/2025` |
| Between | `between 01/03/2025 and 31/03/2025` |
		""")

	col_a, col_b = st.columns([2, 1])
	with col_a:
		period_text = st.text_input(
			"Time period",
			placeholder="e.g., January 2025, last 3 months, Q1 2025",
			help="Describe the period in natural language. The LLM will interpret it.",
		)
	with col_b:
		default_out = os.path.join(os.getcwd(), "invoice_report.xlsx")
		output_path = st.text_input("Output file path", value=default_out)

	if st.button("Generate Report", type="primary"):
		if not period_text.strip():
			st.warning("Please specify a time period.")
		else:
			with st.spinner("Filtering invoices and generating report..."):
				msg = create_invoice_excel_report(DEFAULT_INDEX_PATH, period_text, output_path)
			if "created" in msg.lower():
				st.success(msg)
			else:
				st.warning(msg)


# --------------------------------------------------
# Tab: Manage Invoices
# --------------------------------------------------
with tab_manage:
	st.markdown('<div class="section-header">Manage Invoices</div>', unsafe_allow_html=True)
	st.caption("View detailed information for each invoice and remove records from the index.")

	invoices_with_ids = get_all_invoices_with_ids(DEFAULT_INDEX_PATH)

	if not invoices_with_ids:
		st.info("No invoices in the index. Upload and process PDFs first.")
	else:
		st.markdown(f"**{len(invoices_with_ids)} invoice(s) in index**")
		st.markdown("---")

		for i, inv in enumerate(invoices_with_ids):
			inv_number = inv.get('invoice_number', 'N/A')
			vendor = inv.get('vendor_name', 'Unknown')
			total = inv.get('total_amount', 0)
			date = inv.get('invoice_date', 'N/A')
			status = inv.get('payment_status', 'N/A')
			doc_id = inv.get('_doc_id', '')

			with st.expander(f"{inv_number}  |  {vendor}  |  ${total:,.2f}  |  {date}  |  {status}"):
				col_detail, col_action = st.columns([4, 1])

				with col_detail:
					# Core details
					st.markdown("**Invoice Details**")
					detail_cols = st.columns(3)
					with detail_cols[0]:
						st.text(f"Invoice #: {inv_number}")
						st.text(f"Date: {date}")
						st.text(f"Due Date: {inv.get('due_date', 'N/A')}")
						st.text(f"PO #: {inv.get('purchase_order_number', 'N/A')}")
					with detail_cols[1]:
						st.text(f"Vendor: {vendor}")
						st.text(f"Address: {inv.get('vendor_address', 'N/A')}")
						st.text(f"Contact: {inv.get('vendor_contact', 'N/A')}")
						st.text(f"Tax ID: {inv.get('tax_id', 'N/A')}")
					with detail_cols[2]:
						st.text(f"Buyer: {inv.get('buyer_name', 'N/A')}")
						st.text(f"Buyer Addr: {inv.get('buyer_address', 'N/A')}")
						st.text(f"Status: {status}")
						st.text(f"Method: {inv.get('payment_method', 'N/A')}")

					# Financial summary
					st.markdown("**Financial Summary**")
					fin_cols = st.columns(4)
					with fin_cols[0]:
						st.text(f"Subtotal: ${inv.get('subtotal', 0):,.2f}")
					with fin_cols[1]:
						st.text(f"Tax: ${inv.get('tax', 0):,.2f}")
					with fin_cols[2]:
						st.text(f"Discount: ${inv.get('discount', 0):,.2f}")
					with fin_cols[3]:
						st.text(f"Total: ${total:,.2f} {inv.get('currency', '')}")

					# Line items
					items = inv.get('items', [])
					if items:
						st.markdown("**Line Items**")
						for item in items:
							st.text(f"  - {item}")

				with col_action:
					st.markdown("<br><br>", unsafe_allow_html=True)
					if st.button("Delete", key=f"del_{doc_id}_{i}", type="secondary", use_container_width=True):
						success = delete_invoice(DEFAULT_INDEX_PATH, doc_id)
						if success:
							st.success(f"Deleted {inv_number}")
							st.rerun()
						else:
							st.error("Failed to delete.")


# =============================================
# Footer
# =============================================
st.markdown("""
<div class="app-footer">
	Invoice Automation &mdash; Upload &rarr; Extract &rarr; Query &rarr; Export
</div>
""", unsafe_allow_html=True)
