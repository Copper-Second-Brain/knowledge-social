# backend/tests/test_backend.py
import requests
import json
import sys
import time

BASE_URL = 'http://127.0.0.1:5000/api'

def test_create_document(title, content, author_id='user1', linked_docs=None):
    doc_data = {
        'title': title,
        'content': content,
        'author_id': author_id,
        'linkedDocuments': linked_docs or []
    }
    
    print(f"\nCreating document: {title}")
    print(f"Request data: {json.dumps(doc_data, indent=2)}")
    
    try:
        response = requests.post(
            f'{BASE_URL}/documents', 
            json=doc_data,
            headers={'Content-Type': 'application/json'}
        )
        print('Response Details:')
        print(f'Status Code: {response.status_code}')
        print(f'Content: {response.text}\n')
        
        if response.ok:
            return response.json()['document_id']
        return None
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')
        return None

def test_get_recommendations(user_id='user1'):
    print(f"\nGetting recommendations for user: {user_id}")
    try:
        response = requests.get(f'{BASE_URL}/users/{user_id}/recommendations')
        print('Recommendations Response:')
        print(f'Status Code: {response.status_code}')
        print(f'Content: {response.text}\n')
        return response.json() if response.ok else None
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')
        return None

def test_get_user_graph(user_id='user1'):
    print(f"\nGetting knowledge graph for user: {user_id}")
    try:
        response = requests.get(f'{BASE_URL}/users/{user_id}/graph')
        print('Graph Response:')
        print(f'Status Code: {response.status_code}')
        print(f'Content: {response.text}\n')
        return response.json() if response.ok else None
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')
        return None

def run_tests():
    print("Starting comprehensive backend tests...")
    print(f"Python version: {sys.version}")
    print(f"Requests version: {requests.__version__}")
    print(f"Base URL: {BASE_URL}\n")
    
    # Create a series of related documents
    doc_ids = []
    
    # Document 1: ML Introduction
    doc_id = test_create_document(
        "Introduction to Machine Learning",
        "Machine learning is a subset of artificial intelligence that focuses on data and algorithms."
    )
    if doc_id:
        doc_ids.append(doc_id)
    
    # Document 2: Deep Learning (linked to ML)
    if doc_ids:
        doc_id = test_create_document(
            "Deep Learning Fundamentals",
            "Deep learning is a subset of machine learning using neural networks.",
            linked_docs=doc_ids
        )
        if doc_id:
            doc_ids.append(doc_id)
    
    # Document 3: Neural Networks (linked to both)
    if len(doc_ids) >= 2:
        doc_id = test_create_document(
            "Neural Networks in Practice",
            "Neural networks are the backbone of modern deep learning applications.",
            linked_docs=doc_ids
        )
        if doc_id:
            doc_ids.append(doc_id)
    
    # Small delay to allow async processing
    time.sleep(2)
    
    # Test recommendations
    recommendations = test_get_recommendations()
    
    # Test knowledge graph
    graph = test_get_user_graph()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Documents created: {len(doc_ids)}")
    print(f"Document IDs: {doc_ids}")
    print(f"Recommendations received: {'Yes' if recommendations else 'No'}")
    print(f"Knowledge graph received: {'Yes' if graph else 'No'}")

if __name__ == '__main__':
    run_tests()