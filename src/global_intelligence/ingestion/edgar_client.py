from typing import ClassVar

from edgar import Company, set_identity

from global_intelligence.models import FilingData, FilingMetadata


class EdgarClient:
    FORM_ITEMS: ClassVar[dict[str, list[str]]] = {
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

    def fetch_filing_data(self, ticker: str, form_type: str) -> FilingData:
        company = Company(ticker)
        filing = company.get_filings(form=form_type).latest()

        metadata = FilingMetadata(
            ticker=ticker,
            company_name=filing.company,
            form_type=filing.form,
            report_date=filing.report_date,
            source_url=filing.filing_url,
        )

        filing_obj = filing.obj()
        items = {}

        for item_key in self.FORM_ITEMS[form_type]:
            items[item_key] = filing_obj[item_key]

        return FilingData(metadata=metadata, items=items)

    def get_combined_text(self, data: FilingData) -> str:
        texts = []
        for item_name, item_content in data.items.items():
            texts.append(f"## {item_name}\n\n{item_content}")

        return "\n\n".join(texts)
