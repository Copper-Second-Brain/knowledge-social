import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import networkx as nx
import numpy as np
from ..database.graph_store import GraphStore

class KeGCNv2(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = nn.Linear(in_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, out_dim)
        self.enhance = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # First GCN layer
        x = F.relu(self.conv1(x))
        
        # Second GCN layer
        x = self.conv2(x)
        
        # Knowledge enhancement
        x = x + self.enhance(x)
        
        return x

class RecommendationService:
    def __init__(self, graph_store: GraphStore, embedding_dim: int = 768):
        self.graph_store = graph_store
        self.model = KeGCNv2(embedding_dim, hidden_dim=256, out_dim=128)
        
    def _prepare_graph_data(self, graph: nx.Graph) -> tuple:
        # Convert graph to tensor format
        node_map = {node: idx for idx, node in enumerate(graph.nodes())}
        
        # Prepare node features (you would normally load these from somewhere)
        node_features = torch.randn(len(node_map), 768)  # Placeholder features
        
        # Prepare edge index
        edge_index = []
        for edge in graph.edges():
            edge_index.append([node_map[edge[0]], node_map[edge[1]]])
            edge_index.append([node_map[edge[1]], node_map[edge[0]]])  # Add reverse edge
            
        edge_index = torch.tensor(edge_index).t().contiguous()
        
        return node_features, edge_index, node_map
        
    def compute_user_similarities(self, user_id: str) -> List[Dict]:
        # Get relevant subgraph
        user_graph = self.graph_store.get_subgraph(user_id, depth=2)
        
        # Prepare graph data
        node_features, edge_index, node_map = self._prepare_graph_data(user_graph)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(node_features, edge_index)
        
        # Compute similarities
        user_similarities = []
        user_idx = node_map[user_id]
        user_embedding = embeddings[user_idx]
        
        for node, idx in node_map.items():
            if node != user_id and user_graph.nodes[node]['type'] == 'user':
                similarity = F.cosine_similarity(
                    user_embedding.unsqueeze(0),
                    embeddings[idx].unsqueeze(0)
                ).item()
                
                # Get common topics
                common_topics = self._get_common_topics(user_graph, user_id, node)
                
                user_similarities.append({
                    'user_id': node,
                    'similarity': similarity,
                    'common_topics': common_topics
                })
        
        return sorted(user_similarities, key=lambda x: x['similarity'], reverse=True)
    
    def _get_common_topics(self, graph: nx.Graph, user1_id: str, user2_id: str) -> List[str]:
        user1_docs = set(self.graph_store.get_neighbors(user1_id, edge_type='authorship'))
        user2_docs = set(self.graph_store.get_neighbors(user2_id, edge_type='authorship'))
        
        # Find documents that are connected (similar content)
        common_topics = set()
        for doc1 in user1_docs:
            for doc2 in user2_docs:
                if graph.has_edge(doc1, doc2):
                    common_topics.add(graph.nodes[doc1]['title'])
                    
        return list(common_topics)