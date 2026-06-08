"""Dataset loaders for Czech embedding benchmark sources."""

from dataloaders.czech_text_document import CzechTextDocumentDatasetLoader
from dataloaders.msmarco import MSMarcoDatasetLoader

__all__ = ["CzechTextDocumentDatasetLoader", "MSMarcoDatasetLoader"]
