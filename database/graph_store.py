import networkx as nx
from typing import List, Dict, Optional
from ..models.graph import GraphNode, GraphEdge

class GraphStore:
    def __init__(self):
        self.graph = nx.Graph()
        
    def add_node(self, node: GraphNode) -> None:
        self.graph.add_node(node.id, type=node.type, **node.data)
        
    def add_edge(self, edge: GraphEdge) -> None:
        self.graph.add_edge(edge.source, edge.target, 
                           weight=edge.weight, type=edge.type)
        
    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        neighbors = []
        for neighbor in self.graph.neighbors(node_id):
            if edge_type is None or self.graph[node_id][neighbor]['type'] == edge_type:
                neighbors.append(neighbor)
        return neighbors

    def get_subgraph(self, node_id: str, depth: int = 1) -> nx.Graph:
        return nx.ego_graph(self.graph, node_id, radius=depth)