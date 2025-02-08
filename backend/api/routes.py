# api/routes.py
from flask import Blueprint, request, jsonify
from typing import Dict, Any
import uuid
from ..models.document import Document
from ..services.knowledge_graph import KnowledgeGraphService
from ..services.rag import RAGService
from ..services.recommendation import RecommendationService
import networkx as nx


api = Blueprint('api', __name__)

def init_routes(kg_service: KnowledgeGraphService, 
                rag_service: RAGService,
                recommendation_service: RecommendationService):
    
    @api.route('/documents', methods=['POST'])
    def create_document() -> Dict[str, Any]:
        data = request.json
        doc = Document(
            id=str(uuid.uuid4()),
            title=data['title'],
            content=data['content'],
            author_id=data['author_id']
        )
        
        # Add to knowledge graph
        kg_service.add_document(doc, data['author_id'])
        
        # Add to RAG system and find similar documents
        rag_service.add_document(doc)
        similar_docs = rag_service.find_similar_documents(doc)
        
        # Create automatic links for similar documents
        for similar_doc in similar_docs:
            if similar_doc['similarity'] > 0.7:
                kg_service.link_documents(
                    doc.id,
                    similar_doc['doc_id'],
                    similar_doc['similarity']
                )
        
        # Create manual links
        for linked_doc_id in data.get('linkedDocuments', []):
            kg_service.link_documents(
                doc.id,
                linked_doc_id,
                1.0,
                link_type='reference'
            )
        
        return jsonify({'document_id': doc.id})

    @api.route('/users/<user_id>/recommendations')
    def get_recommendations(user_id: str) -> Dict[str, Any]:
        recommendations = recommendation_service.compute_user_similarities(user_id)
        return jsonify(recommendations)

    @api.route('/users/<user_id>/graph')
    def get_user_graph(user_id: str) -> Dict[str, Any]:
        graph = kg_service.get_user_graph(user_id)
        return jsonify(nx.node_link_data(graph))

    return api