"""Parsing the contents of a pdf file."""


from typing import List
from pathlib import Path
from .IngestorInterface import IngestorInterface
from .QuoteModel import QuoteModel
import subprocess
import random


class PDFIngestor(IngestorInterface):
    """PDFIngestor."""

    allowed_extensions = ['pdf']

    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        """parse."""
        try:
            if not cls.can_ingest(path):
                raise Exception(f'Cannot ingest file with path {path}')

            tmp_file = Path(f'./tmp/{random.randint(0,1000000)}.txt')
            print(tmp_file)
            subprocess.call(['pdftotext', path, str(tmp_file)])

            quotes = []
            with open(tmp_file, "r", encoding='utf8') as file:
                for _, line in enumerate(file):
                    parsed = line.strip().split('-')
                    if parsed != ['']:
                        new_quote = QuoteModel(parsed[0], parsed[1])
                        quotes.append(new_quote)
            return quotes
        except Exception as e:
            print(f'An error occurred while parsing the PDF file: {e}')
            return []
