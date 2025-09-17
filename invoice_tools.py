import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import os
from dotenv import load_dotenv
import pandas as pd
import dateparser

from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Persisted index directory
INDEX_PATH = os.environ.get("INVOICE_INDEX_PATH", "./invoice_index")

# Set up local embedding model to avoid OpenAI API key requirement
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def parse_invoice_content(content: str, filename: str) -> dict:
    """Extract structured data from invoice content"""
    import re
    
    invoice_info = {
        'filename': filename,
        'invoice_number': '',
        'customer': '',
        'date': '',
        'items': [],
        'total': '',
        'raw_text': content  # Keep raw text for debugging
    }
    
    # Handle cases where content is already a single line (from PDF extraction)
    if '\n' not in content:
        # Parse single-line content like: "Invoice #INV-001Customer: Alice JohnsonDate: 2025-09-08ItemQuantityUnit Price TotalLaptop1$1200.00$1200.00Mouse2$25.00$50.00Grand Total$1250.00"
        
        # Extract invoice number
        inv_match = re.search(r'Invoice\s*#?([A-Z0-9-]+)', content, re.IGNORECASE)
        if inv_match:
            invoice_info['invoice_number'] = inv_match.group(1)
        
        # Extract customer
        customer_match = re.search(r'Customer:\s*([A-Za-z\s]+?)(?=Date:|Item|$)', content, re.IGNORECASE)
        if customer_match:
            invoice_info['customer'] = customer_match.group(1).strip()
        
        # Extract date
        date_match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', content)
        if date_match:
            invoice_info['date'] = date_match.group(1)
        
        # Extract grand total
        total_match = re.search(r'Grand\s*Total\s*\$?([0-9,]+\.?\d*)', content, re.IGNORECASE)
        if total_match:
            invoice_info['total'] = f"${total_match.group(1)}"
        
        # Extract items - look for patterns like "ItemNameQuantityUnit Price Total" followed by item lines
        # Find the items section
        items_section_match = re.search(r'Item\s*Quantity\s*Unit\s*Price\s*Total(.*?)Grand\s*Total', content, re.IGNORECASE | re.DOTALL)
        if items_section_match:
            items_text = items_section_match.group(1)
            
            # Split items text and parse each item
            # Look for patterns like "Laptop1$1200.00$1200.00"
            item_pattern = r'([A-Za-z\s]+?)(\d+)\$?([0-9,]+\.?\d*)\$?([0-9,]+\.?\d*)'
            item_matches = re.findall(item_pattern, items_text)
            
            for match in item_matches:
                item_name = match[0].strip()
                quantity = match[1]
                unit_price = f"${match[2]}"
                total_price = f"${match[3]}"
                
                if item_name and len(item_name) > 1:
                    invoice_info['items'].append({
                        'name': item_name,
                        'quantity': quantity,
                        'unit_price': unit_price,
                        'total': total_price
                    })
    
    else:
        # Handle multi-line content (original logic)
        lines = content.split('\n')
        in_items_section = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Extract invoice number
            if 'Invoice #' in line or 'Invoice:' in line:
                if 'Invoice #' in line:
                    invoice_info['invoice_number'] = line.split('Invoice #')[-1].strip()
                else:
                    invoice_info['invoice_number'] = line.split('Invoice:')[-1].strip()
            
            # Extract customer
            elif 'Customer:' in line or 'Client:' in line or 'Bill To:' in line:
                if 'Customer:' in line:
                    invoice_info['customer'] = line.split('Customer:')[-1].strip()
                elif 'Client:' in line:
                    invoice_info['customer'] = line.split('Client:')[-1].strip()
                else:
                    invoice_info['customer'] = line.split('Bill To:')[-1].strip()
            
            # Extract date
            elif 'Date:' in line or 'Invoice Date:' in line:
                if 'Date:' in line:
                    invoice_info['date'] = line.split('Date:')[-1].strip()
                else:
                    invoice_info['date'] = line.split('Invoice Date:')[-1].strip()
            
            # Extract total
            elif 'Grand Total' in line or 'Total:' in line or 'Amount Due:' in line:
                if 'Grand Total' in line:
                    invoice_info['total'] = line.split('Grand Total')[-1].strip()
                elif 'Total:' in line:
                    invoice_info['total'] = line.split('Total:')[-1].strip()
                else:
                    invoice_info['total'] = line.split('Amount Due:')[-1].strip()
            
            # Items section detection
            elif ('Item' in line and 'Quantity' in line) or ('Description' in line and 'Qty' in line) or ('Product' in line and 'Amount' in line):
                in_items_section = True
                continue
            elif in_items_section and line and not any(word in line.lower() for word in ['total', 'subtotal', 'tax', 'grand']):
                # Parse item line
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        item_parts = []
                        quantity_idx = -1
                        
                        for j, part in enumerate(parts):
                            if part.replace('.', '').replace('$', '').replace(',', '').isdigit():
                                quantity_idx = j
                                break
                            else:
                                item_parts.append(part)
                        
                        if quantity_idx > 0 and len(parts) >= quantity_idx + 2:
                            item_name = ' '.join(item_parts)
                            quantity = parts[quantity_idx]
                            unit_price = parts[quantity_idx + 1] if len(parts) > quantity_idx + 1 else ''
                            total_price = parts[quantity_idx + 2] if len(parts) > quantity_idx + 2 else ''
                            
                            if item_name and len(item_name) > 1 and not item_name.isdigit():
                                invoice_info['items'].append({
                                    'name': item_name,
                                    'quantity': quantity,
                                    'unit_price': unit_price,
                                    'total': total_price
                                })
                    except:
                        pass
    
    return invoice_info


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def index_exists() -> bool:
    try:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
        _ = load_index_from_storage(storage_context)
        return True
    except Exception:
        return False


def ingest_pdfs_into_index(pdf_file_paths: List[str]) -> str:
    """
    Ingest the provided PDF files into the LlamaIndex vector store.
    Returns a human-readable summary string.
    """
    ensure_dir(INDEX_PATH)

    documents_to_index: List[Document] = []
    processed = 0

    for file_path in pdf_file_paths:
        if not file_path.lower().endswith(".pdf"):
            continue
        if not os.path.exists(file_path):
            continue
        try:
            reader = SimpleDirectoryReader(input_files=[file_path])
            docs = reader.load_data()
            # Store raw text as a document; extraction can be performed later if needed
            for d in docs:
                documents_to_index.append(
                    Document(text=d.text, metadata={"file_name": os.path.basename(file_path)})
                )
            processed += 1
        except Exception as e:
            # Skip corrupt/unreadable files but continue
            continue

    if not documents_to_index:
        return "No valid PDF files were ingested."

    # Create a fresh index (don't try to load existing one)
    index = VectorStoreIndex.from_documents(documents_to_index)
    index.storage_context.persist(persist_dir=INDEX_PATH)
    return f"Indexed {processed} PDF file(s) into the invoice index."


def load_all_invoice_json() -> List[Dict[str, Any]]:
    """
    Retrieve all nodes' content from the index and parse JSON if possible,
    otherwise wrap raw text under a `raw_text` field.
    """
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=1000)
    nodes = retriever.retrieve("all invoices")

    all_items: List[Dict[str, Any]] = []
    for node in nodes:
        content = node.get_content()
        filename = node.metadata.get('file_name', 'unknown')
        
        # Try to parse the content as structured invoice data
        parsed_data = parse_invoice_content(content, filename)
        if parsed_data and (parsed_data.get('customer') or parsed_data.get('items')):
            all_items.append(parsed_data)
        else:
            # Fallback to raw content if parsing fails
            all_items.append({
                "raw_text": content, 
                "filename": filename,
                "invoice_number": "N/A",
                "customer": "N/A", 
                "date": "N/A",
                "total": "N/A",
                "items": []
            })
    return all_items


def resolve_period(text: str) -> Tuple[datetime.date, datetime.date]:
    text = text.strip().lower()
    
    # Handle date ranges with "to"
    if " to " in text:
        left, right = [p.strip() for p in text.split(" to ", 1)]
        d1 = dateparser.parse(left, settings={"DATE_ORDER": "DMY"})
        d2 = dateparser.parse(right, settings={"DATE_ORDER": "DMY"})
        if d1 and d2:
            return d1.date(), d2.date()

    # Handle natural language month references
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    # Check for month name patterns
    for month_name, month_num in month_names.items():
        if month_name in text:
            # Extract year if mentioned
            import re
            year_match = re.search(r'\b(20\d{2})\b', text)
            year = int(year_match.group(1)) if year_match else datetime.now().year
            
            # Create start and end dates for the month
            first = datetime(year, month_num, 1).date()
            if month_num == 12:
                last = datetime(year + 1, 1, 1).date() - timedelta(days=1)
            else:
                last = datetime(year, month_num + 1, 1).date() - timedelta(days=1)
            return first, last
    
    # Try dateparser for other formats
    parsed = dateparser.parse(text, settings={"DATE_ORDER": "DMY"})
    if parsed:
        start = parsed.date()
        end = start
        if "week" in text:
            monday = start - timedelta(days=start.weekday())
            return monday, monday + timedelta(days=6)
        if "month" in text or (parsed.day == 1 and parsed.hour == 0):
            first = start.replace(day=1)
            next_m = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
            last = next_m - timedelta(days=1)
            return first, last
        return start, end

    raise ValueError(f"Could not interpret time period '{text}'.")


def build_excel_for_period(period_text: str, output_path: str) -> str:
    if not index_exists():
        return "Error: Index does not exist. Please ingest PDFs first."

    all_items = load_all_invoice_json()
    try:
        start_date, end_date = resolve_period(period_text)
        print(f"Debug: Looking for invoices between {start_date} and {end_date}")
    except Exception as e:
        return str(e)

    def parse_invoice_date(value: Any):
        if not value:
            return None
        try:
            # Try different date formats
            date_str = str(value).strip()
            if not date_str or date_str.lower() in ['n/a', 'unknown', '']:
                return None
                
            # Try YYYY-MM-DD format first
            try:
                return datetime.strptime(date_str, "%Y-%m-%d").date()
            except:
                pass
                
            # Try MM/DD/YYYY format
            try:
                return datetime.strptime(date_str, "%m/%d/%Y").date()
            except:
                pass
                
            # Try DD/MM/YYYY format
            try:
                return datetime.strptime(date_str, "%d/%m/%Y").date()
            except:
                pass
                
            # Try dateparser for other formats
            parsed = dateparser.parse(date_str)
            if parsed:
                return parsed.date()
                
        except Exception:
            pass
        return None

    filtered: List[Dict[str, Any]] = []
    debug_info = []
    
    for i, inv in enumerate(all_items):
        # Check multiple possible date field names
        date_fields = ['invoice_date', 'date', 'invoiceDate', 'Date']
        inv_date = None
        
        for field in date_fields:
            if field in inv and inv[field]:
                inv_date = parse_invoice_date(inv[field])
                if inv_date:
                    debug_info.append(f"Invoice {i}: Found date '{inv[field]}' in field '{field}' -> {inv_date}")
                    break
        
        if inv_date and start_date <= inv_date <= end_date:
            filtered.append(inv)
            debug_info.append(f"Invoice {i}: Date {inv_date} is within range")
        elif inv_date:
            debug_info.append(f"Invoice {i}: Date {inv_date} is outside range")
        else:
            debug_info.append(f"Invoice {i}: No valid date found. Available fields: {list(inv.keys())}")

    # Print debug information
    print(f"Debug: Found {len(filtered)} matching invoices out of {len(all_items)} total")
    for info in debug_info[:5]:  # Show first 5 debug messages
        print(f"  {info}")
    if len(debug_info) > 5:
        print(f"  ... and {len(debug_info) - 5} more debug messages")

    if not filtered:
        return f"No invoices found for '{period_text}' (searched {len(all_items)} invoices between {start_date} and {end_date})."

    try:
        pd.DataFrame(filtered).to_excel(output_path, index=False, engine="openpyxl")
        return f"Excel report with {len(filtered)} invoices created for '{period_text}' at '{output_path}'."
    except Exception as e:
        return f"Failed to create the Excel report: {e}"


def ask_invoice_question(question: str) -> str:
    if not index_exists():
        return "Error: Index does not exist. Please ingest PDFs first."
    try:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
        index = load_index_from_storage(storage_context)
        
        # Use LLM to understand the question and generate search queries
        search_queries = generate_search_queries_with_llm(question)
        
        # Search using multiple queries to get comprehensive results
        all_nodes = []
        for query in search_queries:
            retriever = index.as_retriever(similarity_top_k=5)
            nodes = retriever.retrieve(query)
            all_nodes.extend(nodes)
        
        # Remove duplicates
        unique_nodes = []
        seen_content = set()
        for node in all_nodes:
            content = node.get_content()
            if content not in seen_content:
                seen_content.add(content)
                unique_nodes.append(node)
        
        if not unique_nodes:
            return "No relevant information found for your question."
        
        # Use LLM to generate a human-like response from the retrieved content
        return generate_llm_response(question, unique_nodes)
        
    except Exception as e:
        return f"Failed to answer question: {e}"


def generate_search_queries_with_llm(question: str) -> list:
    """Use LLM to generate multiple search queries for better retrieval"""
    try:
        # Use the existing LLM to generate search queries
        from langchain_groq import ChatGroq
        import os
        
        # Get the model from environment or use default
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        # In generate_search_queries_with_llm() and generate_llm_response()

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
        
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=groq_model,
            temperature=0
        )
        
        prompt = f"""
        You are an expert at analyzing questions and generating search queries for document retrieval.
        
        Given this question: "{question}"
        
        Generate 3 different search queries that would help find relevant information in invoice documents.
        The queries should be:
        1. A broad query covering the main topic
        2. A specific query focusing on key terms
        3. An alternative query using different wording
        
        Return only the 3 queries, one per line, without any explanation or numbering.
        
        Examples:
        - If question is "What is quantity of printer we have taken?", generate:
        printer quantity amount
        how many printers purchased
        printer items bought
        
        - If question is "From whom did we buy cars?", generate:
        car purchase vendor customer
        who sold us vehicles
        car supplier company
        """
        
        response = llm.invoke(prompt)
        queries = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
        
        # Fallback if LLM fails
        if not queries:
            queries = [question]
        
        return queries[:3]
        
    except Exception as e:
        # Fallback to original question if LLM fails
        return [question]


def generate_llm_response(question: str, nodes) -> str:
    """Use LLM to generate a human-like response from retrieved content"""
    try:
        # Combine all retrieved content
        all_content = []
        for node in nodes:
            content = node.get_content()
            filename = node.metadata.get('file_name', 'unknown')
            all_content.append(f"From {filename}:\n{content}\n")
        
        combined_content = "\n".join(all_content)
        
        # Use LLM to generate response
        from langchain_groq import ChatGroq
        import os
        
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        # In generate_search_queries_with_llm() and generate_llm_response()

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
        
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=groq_model,
            temperature=0.1
        )
        
        prompt = f"""
        You are a helpful assistant that answers questions about invoice data in a natural, conversational way.
        
        Question: "{question}"
        
        Invoice Data:
        {combined_content}
        
        Based on the invoice data above, answer the question in a natural, human-like way. 
        Be conversational and helpful. If you find relevant information, present it clearly.
        If you don't find the specific information, say so politely.
        
        Guidelines:
        - Use natural language, not technical jargon
        - Be specific about quantities, prices, customers, etc.
        - Mention which invoice/file the information comes from
        - If asking about quantities, show totals and breakdowns
        - If asking about prices, show unit prices and totals
        - If asking about vendors/customers, name them specifically
        - Keep responses concise but informative
        """
        
        response = llm.invoke(prompt)
        return response.content.strip()
        
    except Exception as e:
        # Fallback to template-based response if LLM fails
        return generate_template_response(question, combined_content, nodes)


def generate_template_response(question: str, content: str, nodes) -> str:
    """Fallback template response if LLM fails"""
    return "I couldn't process your question at the moment. Please try again or rephrase your question."


# Removed all hardcoded parsing and response functions
# Now using LLM-powered approach for all responses


# Removed all hardcoded search and response functions
# Now using LLM-powered approach for all responses


def extract_customer_from_content(content: str) -> str:
    """Extract customer name from raw content"""
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if 'Customer:' in line:
            return line.split('Customer:')[-1].strip()
        elif 'Client:' in line:
            return line.split('Client:')[-1].strip()
        elif 'Bill To:' in line:
            return line.split('Bill To:')[-1].strip()
    return None


# Removed all hardcoded parsing and response generation functions
# Now using LLM-powered approach for all responses


__all__ = [
    "INDEX_PATH",
    "ingest_pdfs_into_index",
    "load_all_invoice_json",
    "parse_invoice_content",
    "build_excel_for_period",
    "ask_invoice_question",
    "index_exists",
]


