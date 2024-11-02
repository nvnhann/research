"""Implement abstract base class."""


from abc import ABC, abstractmethod
from typing import List
from .QuoteModel import QuoteModel


class IngestorInterface(ABC):
    """IngestorInterface."""

    allowed_extensions = []

    @classmethod
    def can_ingest(cls, path: str) -> bool:
        """can_ingest."""
        ext = path.split('.')[-1].lower()
        return ext in cls.allowed_extensions

    @classmethod
    @abstractmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        """parse."""
        pass
