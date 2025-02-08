# services/knowledge_graph.py
from typing import List, Dict, Optional
from ..models.document import Document
from ..models.graph import GraphNode, GraphEdge
from ..database.graph_store import GraphStore
import networkx as nx


class KnowledgeGraphService:
    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def add_document(self, doc: Document, user_id: str) -> None:
        # Add document node
        doc_node = GraphNode(
            id=doc.id,
            type='document',
            data={'title': doc.title, 'content': doc.content}
        )
        self.graph_store.add_node(doc_node)

        # Add authorship edge
        auth_edge = GraphEdge(
            source=user_id,
            target=doc.id,
            weight=1.0,
            type='authorship'
        )
        self.graph_store.add_edge(auth_edge)

    def link_documents(self, source_id: str, target_id: str, 
                      similarity: float, link_type: str = 'similarity') -> None:
        edge = GraphEdge(
            source=source_id,
            target=target_id,
            weight=similarity,
            type=link_type
        )
        self.graph_store.add_edge(edge)

    def get_user_graph(self, user_id: str, depth: int = 2) -> nx.Graph:
        return self.graph_store.get_subgraph(user_id, depth)

    def get_document_neighbors(self, doc_id: str) -> List[Dict]:
        neighbors = self.graph_store.get_neighbors(doc_id)
        return [
            {
                'id': neighbor,
                'type': self.graph_store.graph.nodes[neighbor]['type'],
                'weight': self.graph_store.graph[doc_id][neighbor]['weight']
            }
            for neighbor in neighbors
        ]
