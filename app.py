import os
import io
from typing import List

import streamlit as st

from invoice_tools import (
    INDEX_PATH,
    ingest_pdfs_into_index,
    build_excel_for_period,
    ask_invoice_question,
    index_exists,
)


st.set_page_config(page_title="Invoice Assistant", page_icon="📄", layout="wide")


def save_uploaded_files(files: List[st.runtime.uploaded_file_manager.UploadedFile], target_dir: str) -> List[str]:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    saved_paths = []
    for f in files:
        if not f.name.lower().endswith(".pdf"):
            continue
        path = os.path.join(target_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        saved_paths.append(path)
    return saved_paths


def main():
    st.title("📄 Invoice Assistant")
    #st.caption(f"Index path: {INDEX_PATH}")

    tabs = st.tabs(["📥 Ingest PDFs", "📋 View PDFs", "📊 Build Excel", "❓ Ask a Question"]) 

    with tabs[0]:
        st.subheader("Ingest PDF invoices into index")
        upload_dir = st.text_input("Upload target folder", value="invoices_to_process")
        files = st.file_uploader("Upload PDF invoices", type=["pdf"], accept_multiple_files=True)
        if st.button("Ingest to Index", type="primary"):
            if not files:
                st.warning("Please upload at least one PDF.")
            else:
                saved = save_uploaded_files(files, upload_dir)
                msg = ingest_pdfs_into_index(saved)
                st.success(msg)

    with tabs[1]:
        st.subheader("View uploaded PDFs and their content")
        if not index_exists():
            st.info("No index found. Please ingest PDFs on the first tab.")
        else:
            try:
                from invoice_tools import load_all_invoice_json
                invoice_data = load_all_invoice_json()
                
                if not invoice_data:
                    st.warning("No invoice data found in the index.")
                else:
                    st.success(f"Found {len(invoice_data)} invoices in the index")
                    
                    # Show summary
                    st.subheader("📊 Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Invoices", len(invoice_data))
                    with col2:
                        total_items = sum(len(inv.get('items', [])) for inv in invoice_data)
                        st.metric("Total Items", total_items)
                    with col3:
                        customers = set(inv.get('customer', 'Unknown') for inv in invoice_data)
                        st.metric("Unique Customers", len(customers))
                    
                    # Show detailed view
                    st.subheader("📋 Invoice Details")
                    for i, invoice in enumerate(invoice_data):
                        with st.expander(f"📄 {invoice.get('filename', 'Unknown')} - {invoice.get('customer', 'Unknown Customer')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Invoice Number:** {invoice.get('invoice_number', 'N/A')}")
                                st.write(f"**Customer:** {invoice.get('customer', 'N/A')}")
                                st.write(f"**Date:** {invoice.get('date', 'N/A')}")
                                st.write(f"**Total:** {invoice.get('total', 'N/A')}")
                            with col2:
                                st.write(f"**Items ({len(invoice.get('items', []))}):**")
                                for item in invoice.get('items', []):
                                    st.write(f"• {item.get('name', 'N/A')} - Qty: {item.get('quantity', 'N/A')} - ${item.get('unit_price', 'N/A')}")
                            
                            # Show raw content for debugging
                            if st.checkbox(f"Show raw content for {invoice.get('filename', 'Unknown')}", key=f"raw_{i}"):
                                st.text_area("Raw PDF Content", invoice.get('raw_text', 'No raw text available'), height=200, key=f"raw_content_{i}")
                                
            except Exception as e:
                st.error(f"Error loading invoice data: {e}")

    with tabs[2]:
        st.subheader("Create Excel report")
        st.write("Provide a time period (e.g., 'August 2025', 'last month', '21/03/2025 to 21/06/2025').")
        period = st.text_input("Period", value="August 2025")
        output = st.text_input("Output Excel Path", value="invoice_summary.xlsx")
        if st.button("Generate Excel", type="primary"):
            if not index_exists():
                st.info("No index found. Please ingest PDFs on the first tab.")
            else:
                result = build_excel_for_period(period, output)
                if result.startswith("Excel report") and os.path.exists(output):
                    st.success(result)
                    with open(output, "rb") as f:
                        st.download_button("Download Excel", f, file_name=os.path.basename(output))
                else:
                    st.warning(result)

    with tabs[3]:
        st.subheader("Ask about your invoices")
        q = st.text_input("Question", value="What quantity did we buy from XYZ Vendor?")
        if st.button("Ask", type="primary"):
            if not index_exists():
                st.info("No index found. Please ingest PDFs on the first tab.")
            else:
                answer = ask_invoice_question(q)
                st.write("**Answer:**")
                st.write(answer)


if __name__ == "__main__":
    main()


