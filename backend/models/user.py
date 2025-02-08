from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from .document import Document

@dataclass
class User:
    id: str
    username: str
    email: str
    created_at: datetime = datetime.now()
    documents: List[Document] = None
    knowledge_graph: Dict = None

    def __post_init__(self):
        self.documents = self.documents or []
        self.knowledge_graph = self.knowledge_graph or {}




