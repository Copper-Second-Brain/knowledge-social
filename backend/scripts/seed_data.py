# backend/scripts/seed_data.py
import requests
import json
import time
from typing import Dict, List

BASE_URL = 'http://127.0.0.1:5000/api'

class DataSeeder:
    def __init__(self):
        self.document_ids: Dict[str, List[str]] = {}  # user_id -> list of their doc_ids
        
    def create_document(self, title: str, content: str, author_id: str, linked_docs: List[str] = None) -> str:
        doc_data = {
            'title': title,
            'content': content,
            'author_id': author_id,
            'linkedDocuments': linked_docs or []
        }
        
        try:
            response = requests.post(
                f'{BASE_URL}/documents',
                json=doc_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.ok:
                doc_id = response.json()['document_id']
                if author_id not in self.document_ids:
                    self.document_ids[author_id] = []
                self.document_ids[author_id].append(doc_id)
                print(f"Created document: {title} (ID: {doc_id})")
                return doc_id
            else:
                print(f"Failed to create document: {title}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error creating document: {e}")
            return None

    def seed_data(self):
        print("Starting to seed data...")
        
        # User 1: ML Expert
        user1_id = "user1"
        print(f"\nCreating documents for {user1_id} (ML Expert)")
        
        doc1 = self.create_document(
            "Introduction to Machine Learning",
            """Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.
            Key concepts include supervised learning, unsupervised learning, and reinforcement learning.""",
            user1_id
        )
        
        time.sleep(1)  # Small delay between requests
        
        doc2 = self.create_document(
            "Deep Learning Fundamentals",
            """Deep learning is a subset of machine learning focusing on artificial neural networks with multiple layers.
            This approach has revolutionized computer vision and natural language processing.""",
            user1_id,
            linked_docs=[doc1] if doc1 else []
        )
        
        # User 2: Data Science Specialist
        user2_id = "user2"
        print(f"\nCreating documents for {user2_id} (Data Science Specialist)")
        
        doc3 = self.create_document(
            "Data Science Best Practices",
            """Data science combines statistics, programming, and domain expertise to extract insights from data.
            Essential skills include data cleaning, visualization, and statistical analysis.""",
            user2_id
        )
        
        time.sleep(1)
        
        doc4 = self.create_document(
            "Advanced Data Visualization",
            """Effective data visualization is crucial for communicating insights. This guide covers advanced techniques
            using popular tools like matplotlib, seaborn, and plotly.""",
            user2_id,
            linked_docs=[doc3] if doc3 else []
        )
        
        # User 3: AI Researcher
        user3_id = "user3"
        print(f"\nCreating documents for {user3_id} (AI Researcher)")
        
        doc5 = self.create_document(
            "AI Alignment Principles",
            """AI alignment focuses on ensuring artificial intelligence systems behave in accordance with human values.
            Key considerations include safety, ethics, and robustness.""",
            user3_id
        )
        
        time.sleep(1)
        
        doc6 = self.create_document(
            "LSTM Networks Explained",
            """Long Short-Term Memory networks are a type of RNN capable of learning long-term dependencies.
            They are particularly effective for sequential data like text and time series.""",
            user3_id,
            linked_docs=[doc2, doc5] if doc2 and doc5 else []
        )
        
        # User 4: NLP Specialist
        user4_id = "user4"
        print(f"\nCreating documents for {user4_id} (NLP Specialist)")
        
        doc7 = self.create_document(
            "Natural Language Processing Fundamentals",
            """NLP combines linguistics and machine learning to enable computers to understand human language.
            Core concepts include tokenization, parsing, and semantic analysis.""",
            user4_id
        )
        
        time.sleep(1)
        
        doc8 = self.create_document(
            "Advanced Language Models",
            """Modern language models use transformer architectures to achieve state-of-the-art performance.
            This document covers attention mechanisms and their applications.""",
            user4_id,
            linked_docs=[doc6, doc7] if doc6 and doc7 else []
        )

        print("\nSeeding completed!")
        print("\nDocument IDs by user:")
        for user_id, doc_ids in self.document_ids.items():
            print(f"{user_id}: {doc_ids}")

if __name__ == "__main__":
    seeder = DataSeeder()
    seeder.seed_data()