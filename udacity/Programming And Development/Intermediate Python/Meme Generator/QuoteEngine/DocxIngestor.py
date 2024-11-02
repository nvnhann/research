"""Parsing the contents of a docx file."""


from typing import List
from .IngestorInterface import IngestorInterface
from .QuoteModel import QuoteModel
import docx


class DocxIngestor(IngestorInterface):
    """DocxIngestor."""

    allowed_extensions = ['docx']

    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        """parse."""
        if not cls.can_ingest(path):
            raise Exception(f"Cannot ingest {path}.")

        quotes = []
        try:
            document = docx.Document(path)

            for paragraph in document.paragraphs:
                text = paragraph.text.strip()
                if text:
                    author, body = map(str.strip, text.split("-"))
                    quote = QuoteModel(body, author)
                    quotes.append(quote)
        except Exception as e:
            raise Exception(f"Error parsing {path}: {e}")

        return quotes
