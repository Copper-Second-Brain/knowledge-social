from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from ..models.document import Document
from ..database.vector_store import VectorStore

class RAGService:
    def __init__(self, vector_store: VectorStore):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = vector_store
        
    def add_document(self, doc: Document) -> None:
        """Add document to RAG system"""
        if doc.embedding is None:
            doc.embedding = self.model.encode(doc.content)
        self.vector_store.add_document(doc)
        
    def find_similar_documents(self, doc: Document, k: int = 5) -> List[Dict]:
        """Find similar documents using RAG"""
        if doc.embedding is None:
            doc.embedding = self.model.encode(doc.content)
            
        similar_docs = self.vector_store.search(doc.embedding, k)
        return [
            {
                'doc_id': doc_id,
                'similarity': similarity
            }
            for doc_id, similarity in similar_docs
            if doc_id != doc.id  # Exclude the query document itself
        ]