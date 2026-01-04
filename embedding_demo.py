#!/usr/bin/env python3
"""
Vector Embeddings Demo with VoyageAI
====================================

This script demonstrates how text documents are converted into vector embeddings
using VoyageAI's embedding models. Vector embeddings are numerical representations
of text that capture semantic meaning and enable similarity searches.
"""

import os
import voyageai
from dotenv import load_dotenv
import numpy as np
from typing import List

# Load environment variables
load_dotenv()

def initialize_voyage_client():
    """Initialize the VoyageAI client."""
    api_key = os.getenv("VOYAGEAI_API_KEY")
    if not api_key:
        raise ValueError("VOYAGEAI_API_KEY not found in environment variables")
    
    return voyageai.Client(api_key=api_key)

def display_document(title: str, content: str):
    """Display a document in a formatted way."""
    print(f"\n{'='*60}")
    print(f"📄 {title}")
    print(f"{'='*60}")
    print(f"Content: {content}")
    print(f"Length: {len(content)} characters")
    print(f"Word count: {len(content.split())} words")

def create_embeddings(client, documents: List[str], model: str = "voyage-3") -> List[List[float]]:
    """Create embeddings for a list of documents."""
    print(f"\n🚀 Creating embeddings using model: {model}")
    print("Sending documents to VoyageAI...")
    
    try:
        # Create embeddings
        result = client.embed(documents, model=model)
        embeddings = result.embeddings
        
        print(f"✅ Successfully created {len(embeddings)} embeddings")
        return embeddings
    
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}")
        return []

def analyze_embedding(embedding: List[float], doc_title: str):
    """Analyze and display embedding properties."""
    embedding_array = np.array(embedding)
    
    print(f"\n🔍 Analyzing embedding for: {doc_title}")
    print(f"{'='*50}")
    print(f"Dimension: {len(embedding)} (vector size)")
    
    # Show first 10 values as example
    print(f"\nFirst 10 values: {embedding[:10]}")
    print(f"Last 10 values: {embedding[-10:]}")

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)

def main():
    """Main demonstration function."""
    print("🎯 Vector Embeddings Demo with VoyageAI")
    print("=" * 60)
    print("This demo shows how text is converted into vector embeddings")
    print("that capture semantic meaning for AI applications.")
    
    try:
        # Initialize VoyageAI client
        client = initialize_voyage_client()
        print("✅ VoyageAI client initialized successfully")
        
        # Sample document to embed
        document = "Machine learning algorithms can process vast amounts of data to identify patterns and make predictions. This technology enables computers to learn from experience and make intelligent decisions without being explicitly programmed for every scenario."
        documents = [document]
        
        # Display the document
        display_document("Sample Document", document)
        
        # Create embeddings for all documents
        embeddings = create_embeddings(client, documents, model="voyage-3")
        
        if not embeddings:
            print("❌ Failed to create embeddings. Exiting...")
            return
        
        # Analyze the embedding
        embedding = embeddings[0]
        
        # Show vector properties
        print(f"\n🔍 Vector Properties")
        print(f"{'='*60}")
        print("This vector embedding captures the semantic meaning of the text.")
        print("Each dimension represents different aspects of the content.")
        print()
        
        # Show what the raw embedding looks like (truncated for readability)
        print(f"\n📊 Raw Embedding Example")
        print(f"{'='*60}")
        print("This is what the actual vector looks like (showing first 20 values):")
        embedding_sample = embedding[:20]
        formatted_values = [f"{val:.6f}" for val in embedding_sample]
        print(f"[{', '.join(formatted_values)}, ...]")
        print(f"\n💡 This vector has {len(embedding)} dimensions total!")
        print("Each dimension captures different semantic features of the text.")
            
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        return

if __name__ == "__main__":
    main()