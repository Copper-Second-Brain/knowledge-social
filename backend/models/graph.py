from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class GraphNode:
    id: str
    type: str  # 'user' or 'document'
    data: Dict[str, Any]

@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float
    type: str  # 'authorship', 'reference', or 'similarity'

