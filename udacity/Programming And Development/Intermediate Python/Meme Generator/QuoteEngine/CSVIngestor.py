"""Parsing the contents of a csv file."""


from typing import List
from .IngestorInterface import IngestorInterface
from .QuoteModel import QuoteModel
import pandas as pd


class CSVIngestor(IngestorInterface):
    """CSVIngestor."""

    allowed_extensions = ['csv']

    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        """parse."""
        if not cls.can_ingest(path):
            raise ValueError(f"Cannot ingest file at {path}")

        quotes = []

        try:
            df = pd.read_csv(path, header=0)
        except pd.errors.EmptyDataError:
            # Handle the case where the csv file is empty
            print(f"Empty csv file at {path}")
            return quotes
        except pd.errors.ParserError:
            # Handle the case where there is an issue parsing the csv file
            print(f"Error parsing csv file at {path}")
            return quotes

        for _, row in df.iterrows():
            quote = QuoteModel(row['body'], row['author'])
            quotes.append(quote)
        return quotes
