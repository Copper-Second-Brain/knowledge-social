# models/document.py
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class Document:
    id: str
    title: str
    content: str
    author_id: str
    created_at: datetime = datetime.now()
    links: List[str] = None  # Document IDs this document links to
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self.links = self.links or []