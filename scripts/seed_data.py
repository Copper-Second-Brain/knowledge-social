# backend/scripts/enhanced_seed_data.py
import requests
import json
import time
from typing import Dict, List

BASE_URL = 'http://127.0.0.1:5000/api'

class EnhancedDataSeeder:
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
        print("Starting to seed enhanced data...")
        
        # User 1: ML Expert
        user1_id = "user1"
        print(f"\nCreating documents for {user1_id} (ML Expert)")
        
        doc1 = self.create_document(
            "Introduction to Machine Learning",
            """Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.
            Key concepts include supervised learning, unsupervised learning, and reinforcement learning.
            
            In supervised learning, algorithms learn from labeled training data, making predictions or decisions based on that data.
            Common supervised learning algorithms include linear regression, logistic regression, support vector machines, and neural networks.
            
            Unsupervised learning works with unlabeled data, identifying patterns or intrinsic structures within the data.
            Clustering and dimensionality reduction are common applications of unsupervised learning.""",
            user1_id
        )
        
        time.sleep(1)  # Small delay between requests
        
        doc2 = self.create_document(
            "Deep Learning Fundamentals",
            """Deep learning is a subset of machine learning focusing on artificial neural networks with multiple layers.
            This approach has revolutionized computer vision and natural language processing.
            
            Neural networks are inspired by the human brain, consisting of interconnected nodes (neurons) organized in layers.
            The "deep" in deep learning refers to the use of multiple hidden layers in the neural network.
            
            Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images.
            They use convolutional layers to detect features and pooling layers to reduce dimensionality.
            
            Recurrent Neural Networks (RNNs) are designed for sequential data, maintaining internal state (memory) to process sequences.
            Long Short-Term Memory (LSTM) networks are a type of RNN that solve the vanishing gradient problem.""",
            user1_id,
            linked_docs=[doc1] if doc1 else []
        )
        
        time.sleep(1)
        
        doc3 = self.create_document(
            "Reinforcement Learning Explained",
            """Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.
            
            Unlike supervised learning, reinforcement learning does not require labeled input/output pairs. Instead, the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge).
            
            Key concepts in reinforcement learning include:
            - Agent: The learner or decision maker
            - Environment: Everything the agent interacts with
            - State: Current situation of the agent
            - Action: Choices that the agent can make
            - Reward: Feedback from the environment
            
            Popular reinforcement learning algorithms include Q-learning, Deep Q-Networks (DQN), and Policy Gradient methods.""",
            user1_id,
            linked_docs=[doc1, doc2] if doc1 and doc2 else []
        )
        
        # User 2: Data Science Specialist
        user2_id = "user2"
        print(f"\nCreating documents for {user2_id} (Data Science Specialist)")
        
        doc4 = self.create_document(
            "Data Science Best Practices",
            """Data science combines statistics, programming, and domain expertise to extract insights from data.
            Essential skills include data cleaning, visualization, and statistical analysis.
            
            The data science workflow typically involves:
            1. Problem Definition: Clearly articulate the question you're trying to answer
            2. Data Collection: Gather relevant data from appropriate sources
            3. Data Cleaning: Handle missing values, outliers, and inconsistencies
            4. Exploratory Data Analysis: Understand patterns and relationships in the data
            5. Feature Engineering: Create new features to improve model performance
            6. Modeling: Build and evaluate predictive models
            7. Communication: Present findings in an understandable way
            
            Effective data scientists combine technical skills with domain knowledge and communication abilities.""",
            user2_id
        )
        
        time.sleep(1)
        
        doc5 = self.create_document(
            "Advanced Data Visualization",
            """Effective data visualization is crucial for communicating insights and findings to stakeholders.
            
            When creating visualizations, consider these principles:
            - Choose the right chart type for your data and message
            - Minimize clutter and maximize the data-to-ink ratio
            - Use color strategically to highlight important points
            - Label elements directly rather than using legends when possible
            - Ensure accessibility for all viewers
            
            Popular visualization libraries in Python include Matplotlib, Seaborn, and Plotly.
            For interactive visualizations, consider D3.js (for web) or Tableau (for business intelligence).
            
            Advanced techniques like small multiples, heat maps, and network graphs can effectively communicate complex relationships in your data.""",
            user2_id,
            linked_docs=[doc4] if doc4 else []
        )
        
        time.sleep(1)
        
        doc6 = self.create_document(
            "Statistical Methods for Data Science",
            """Statistics forms the foundation of data science, providing the mathematical framework for understanding data.
            
            Key statistical concepts include:
            - Descriptive statistics: Summarizing and describing data features (mean, median, standard deviation)
            - Inferential statistics: Drawing conclusions and making predictions from data
            - Probability distributions: Models describing the likelihood of different outcomes
            - Hypothesis testing: Evaluating whether observed effects are statistically significant
            - Regression analysis: Modeling relationships between variables
            
            Bayesian statistics offers an alternative framework to traditional (frequentist) approaches, updating beliefs based on new evidence.
            Modern computational methods like bootstrapping and permutation tests provide powerful alternatives when traditional assumptions aren't met.""",
            user2_id,
            linked_docs=[doc4, doc5] if doc4 and doc5 else []
        )
        
        # User 3: AI Researcher
        user3_id = "user3"
        print(f"\nCreating documents for {user3_id} (AI Researcher)")
        
        doc7 = self.create_document(
            "AI Alignment Principles",
            """AI alignment focuses on ensuring artificial intelligence systems behave in accordance with human values.
            Key considerations include safety, ethics, and robustness.
            
            As AI systems become more capable, ensuring they remain aligned with human intentions becomes increasingly important.
            Alignment challenges include:
            - Specification problems: Difficulty in precisely defining human values and goals
            - Robustness to distribution shift: Ensuring AI behaves well in new situations
            - Scalable oversight: Maintaining control as systems become more complex
            - Power-seeking behavior: Preventing emergent incentives for systems to resist correction
            
            Current approaches to alignment include reward modeling, constitutional AI, and interpretability research.
            Creating safe and beneficial AI requires interdisciplinary collaboration between technical researchers and ethicists.""",
            user3_id
        )
        
        time.sleep(1)
        
        doc8 = self.create_document(
            "LSTM Networks Explained",
            """Long Short-Term Memory networks are a type of RNN capable of learning long-term dependencies.
            They are particularly effective for sequential data like text and time series.
            
            Traditional RNNs struggle with the vanishing gradient problem, making it difficult to learn long-range dependencies.
            LSTMs address this with a specialized cell structure featuring:
            - A memory cell that can maintain information over long periods
            - An input gate controlling when new information enters the memory
            - A forget gate allowing the network to discard irrelevant information
            - An output gate determining what information is used for predictions
            
            Common LSTM applications include language modeling, speech recognition, and time series forecasting.
            Variations like bidirectional LSTMs and Gated Recurrent Units (GRUs) offer different trade-offs in complexity and performance.""",
            user3_id,
            linked_docs=[doc2, doc7] if doc2 and doc7 else []
        )
        
        time.sleep(1)
        
        doc9 = self.create_document(
            "Transformer Architecture Deep Dive",
            """The Transformer architecture revolutionized natural language processing through its self-attention mechanism.
            
            Unlike recurrent networks, Transformers process entire sequences simultaneously, enabling more efficient training.
            Key components include:
            - Multi-head self-attention: Allowing the model to focus on different parts of the input simultaneously
            - Positional encodings: Preserving sequence order information
            - Layer normalization: Stabilizing training across deep networks
            - Feed-forward networks: Processing contextualized representations
            
            The success of Transformers led to breakthrough models like BERT, GPT, and T5, which achieve state-of-the-art results on diverse language tasks through pre-training and fine-tuning.
            Scaling laws suggest that Transformer performance continues to improve with model size, dataset size, and compute.""",
            user3_id,
            linked_docs=[doc2, doc8] if doc2 and doc8 else []
        )
        
        # User 4: NLP Specialist
        user4_id = "user4"
        print(f"\nCreating documents for {user4_id} (NLP Specialist)")
        
        doc10 = self.create_document(
            "Natural Language Processing Fundamentals",
            """NLP combines linguistics and machine learning to enable computers to understand human language.
            Core concepts include tokenization, parsing, and semantic analysis.
            
            The NLP pipeline typically includes:
            - Preprocessing: Tokenization, stemming/lemmatization, and stop word removal
            - Syntactic analysis: Part-of-speech tagging and parsing to understand sentence structure
            - Semantic analysis: Extracting meaning from text through techniques like named entity recognition
            - Pragmatic analysis: Understanding context and intention beyond literal meaning
            
            Traditional NLP approaches used hand-crafted features and statistical methods, while modern approaches leverage deep learning for end-to-end solutions.
            Challenges in NLP include handling ambiguity, sarcasm, cultural references, and domain-specific terminology.""",
            user4_id
        )
        
        time.sleep(1)
        
        doc11 = self.create_document(
            "Advanced Language Models",
            """Modern language models use transformer architectures to achieve state-of-the-art performance.
            This document covers attention mechanisms and their applications.
            
            Large language models like GPT, BERT, and their successors have transformed the NLP landscape by demonstrating:
            - Transfer learning: Pre-training on large corpora then fine-tuning for specific tasks
            - Few-shot learning: Solving tasks with minimal task-specific examples
            - Emergent abilities: Capabilities that appear only at sufficient scale
            
            The scaling of these models has led to remarkable capabilities in text generation, summarization, translation, and even reasoning tasks.
            However, they also face challenges including hallucination, bias, and lack of controllability.
            
            Current research focuses on improving efficiency, alignment with human values, and factuality in model outputs.""",
            user4_id,
            linked_docs=[doc9, doc10] if doc9 and doc10 else []
        )
        
        time.sleep(1)
        
        doc12 = self.create_document(
            "Prompt Engineering Techniques",
            """Prompt engineering is the practice of designing and optimizing input prompts to effectively guide language model behavior.
            
            Effective prompt engineering strategies include:
            - Chain-of-thought prompting: Encouraging step-by-step reasoning
            - Few-shot learning: Providing examples within the prompt
            - Role prompting: Assigning a specific role to the model
            - Instruction refinement: Clearly articulating desired outputs
            
            These techniques can dramatically improve performance on complex tasks without modifying the underlying model.
            The emergence of prompt engineering highlights the importance of interaction design in AI systems.
            
            Advanced techniques like retrieval-augmented generation combine prompting with external knowledge sources to improve factuality and reduce hallucination.""",
            user4_id,
            linked_docs=[doc10, doc11] if doc10 and doc11 else []
        )
        
        print("\nEnhanced data seeding completed!")
        print("\nDocument IDs by user:")
        for user_id, doc_ids in self.document_ids.items():
            print(f"{user_id}: {doc_ids}")

if __name__ == "__main__":
    seeder = EnhancedDataSeeder()
    seeder.seed_data()