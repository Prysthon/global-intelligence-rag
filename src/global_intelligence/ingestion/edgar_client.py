from typing import Any

from edgar import Company, set_identity


class EdgarClient:
    FORM_ITEMS = {  # noqa: RUF012
        "10-K": ["Item 1", "Item 1A", "Item 7", "Item 8", "Item 9A"],
        "10-Q": [
            "Part I, Item 1",
            "Part I, Item 2",
            "Part I, Item 3",
            "Part I, Item 4",
        ],
    }

    def __init__(self, email: str):
        set_identity(email)

    def fetch_filing_data(self, ticker: str, form_type: str) -> dict[str, Any]:
        company = Company(ticker)
        filing = company.get_filings(form=form_type).latest()

        metadata = {
            "ticker": ticker,
            "company_name": filing.company,
            "form_type": filing.form,
            "report_date": str(filing.report_date),
            "source_url": filing.filing_url,
        }

        filing_obj = filing.obj()
        items = {}

        for item_key in self.FORM_ITEMS[form_type]:
            items[item_key] = filing_obj[item_key]

        return {"metadata": metadata, "items": items}

    def get_combined_text(self, data: dict[str, Any]) -> str:
        texts = []
        for item_name, item_content in data["items"].items():
            texts.append(f"## {item_name}\n\n{item_content}")

        return "\n\n".join(texts)
