# backend/__init__.py
from flask import Flask
from flask_cors import CORS
from .database.vector_store import VectorStore
from .database.graph_store import GraphStore
from .services.knowledge_graph import KnowledgeGraphService
from .services.rag import RAGService
from .services.recommendation import RecommendationService
from .api.routes import init_routes

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Initialize databases
    vector_store = VectorStore()
    graph_store = GraphStore()

    # Initialize services
    kg_service = KnowledgeGraphService(graph_store)
    rag_service = RAGService(vector_store)
    recommendation_service = RecommendationService(graph_store)

    # Register routes
    api = init_routes(kg_service, rag_service, recommendation_service)
    app.register_blueprint(api, url_prefix='/api')

    return app





