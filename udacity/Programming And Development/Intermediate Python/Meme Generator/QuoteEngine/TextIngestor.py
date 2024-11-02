"""Parsing the contents of a text file."""


from typing import List
from .IngestorInterface import IngestorInterface
from .QuoteModel import QuoteModel


class TextIngestor(IngestorInterface):
    """TextIngestor."""

    allowed_extensions = ['txt']

    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        """parse."""
        if not cls.can_ingest(path):
            raise ValueError('Unsupported file extension')

        quotes = []

        try:
            with open(path, 'r', encoding='utf8') as file:
                for line in file:
                    parts = line.strip().split('-')
                    if len(parts) == 2:
                        author, body = parts
                        quote = QuoteModel(author.strip(), body.strip())
                        quotes.append(quote)
        except FileNotFoundError:
            print(f"File '{path}' not found")

        return quotes
