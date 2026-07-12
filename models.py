"""
Pydantic data models for structured invoice extraction.
"""

from pydantic import BaseModel, Field


class InvoiceDetails(BaseModel):
	"""Schema for structured data extracted from a single invoice PDF."""

	vendor_name: str = Field(description="The name of the vendor or company.")
	vendor_address: str = Field(description="Address of the vendor or company.")
	vendor_contact: str = Field(description="Contact phone number or email of the vendor, if present. Otherwise empty string.")
	tax_id: str = Field(description="The VAT, GST, or Tax registration identifier of the vendor, if present. Otherwise empty string.")
	buyer_name: str = Field(description="Name of the buyer or client.")
	buyer_address: str = Field(description="Address of the buyer or client.")
	invoice_number: str = Field(description="The unique invoice identifier.")
	invoice_date: str = Field(description="The date of the invoice (YYYY-MM-DD).")
	due_date: str = Field(description="The payment due date of the invoice (YYYY-MM-DD), if present. Otherwise empty string.")
	payment_status: str = Field(description="The current payment status of the invoice, e.g., 'Paid', 'Unpaid', 'Overdue', 'Partially Paid'. If not specified, default to 'Unpaid'.")
	payment_method: str = Field(description="The payment method used or specified (e.g., 'Bank Transfer', 'Credit Card', 'Bank Check', 'Cash'), if present. Otherwise empty string.")
	purchase_order_number: str = Field(description="The purchase order (PO) number associated with the invoice, if present. Otherwise empty string.")
	currency: str = Field(description="The currency of the invoice amounts (e.g., 'USD', 'EUR', 'INR', 'GBP'). If not clear, default to 'USD'.")
	items: list[str] = Field(description="List of purchased items with details.")
	subtotal: float = Field(description="Subtotal before tax and discount.")
	tax: float = Field(description="Tax applied to the invoice.")
	discount: float = Field(description="Discount applied to the invoice.")
	total_amount: float = Field(description="The final total amount due after tax and discount.")
