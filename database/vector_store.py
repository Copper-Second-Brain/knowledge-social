import numpy as np
import faiss
from typing import List, Dict, Tuple
from ..models.document import Document

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.doc_map: Dict[int, str] = {}  # Maps faiss index to document ID
        self.current_index = 0

    def add_document(self, doc: Document) -> None:
        if doc.embedding is None:
            raise ValueError("Document must have embedding before adding to vector store")
        
        self.index.add(np.array([doc.embedding]))
        self.doc_map[self.current_index] = doc.id
        self.current_index += 1

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        D, I = self.index.search(np.array([query_embedding]), k)
        results = []
        for idx, distance in zip(I[0], D[0]):
            if idx in self.doc_map:
                similarity = 1 - (distance / 2)  # Convert distance to similarity
                results.append((self.doc_map[idx], similarity))
        return results