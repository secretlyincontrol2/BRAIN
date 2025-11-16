"""
Example usage of the BRAIN embedding system
"""
import torch
from PIL import Image
import numpy as np
import requests
from pathlib import Path
import matplotlib.pyplot as plt

from model import SwinEmbedder
from faiss_retrieval import FAISSIndex
from transformers import AutoImageProcessor
from config import Config


def example_1_generate_embedding():
    """Example 1: Generate embedding for a single image"""
    print("\n" + "="*50)
    print("Example 1: Generate Embedding")
    print("="*50)
    
    # Load model
    print("Loading model...")
    model = SwinEmbedder()
    model.eval()
    
    # Load processor
    processor = AutoImageProcessor.from_pretrained(
        Config.MODEL_NAME,
        cache_dir=str(Config.MODEL_DIR)
    )
    
    # Create a dummy image (replace with real image path)
    print("Creating dummy image...")
    dummy_image = Image.new('RGB', (224, 224), color='white')
    
    # Process image
    inputs = processor(images=dummy_image, return_tensors="pt")
    
    # Generate embedding
    print("Generating embedding...")
    with torch.no_grad():
        embedding, _ = model(inputs['pixel_values'])
    
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Embedding dimension: {embedding.shape[1]}")
    print(f"✓ Embedding norm: {torch.norm(embedding).item():.4f}")
    print(f"✓ First 5 values: {embedding[0, :5].tolist()}")


def example_2_batch_embeddings():
    """Example 2: Generate embeddings for multiple images"""
    print("\n" + "="*50)
    print("Example 2: Batch Embeddings")
    print("="*50)
    
    # Load model
    model = SwinEmbedder()
    model.eval()
    
    processor = AutoImageProcessor.from_pretrained(
        Config.MODEL_NAME,
        cache_dir=str(Config.MODEL_DIR)
    )
    
    # Create dummy images
    num_images = 4
    dummy_images = [Image.new('RGB', (224, 224), color=(i*50, i*50, i*50)) 
                   for i in range(num_images)]
    
    # Process batch
    inputs = processor(images=dummy_images, return_tensors="pt")
    
    # Generate embeddings
    print(f"Generating embeddings for {num_images} images...")
    with torch.no_grad():
        embeddings, _ = model(inputs['pixel_values'])
    
    print(f"✓ Embeddings shape: {embeddings.shape}")
    
    # Calculate pairwise distances
    distances = torch.cdist(embeddings, embeddings)
    print(f"✓ Pairwise distances:\n{distances}")


def example_3_faiss_retrieval():
    """Example 3: FAISS similarity search"""
    print("\n" + "="*50)
    print("Example 3: FAISS Retrieval")
    print("="*50)
    
    # Create a small FAISS index
    print("Creating FAISS index...")
    faiss_index = FAISSIndex(embedding_dim=768)
    
    # Generate dummy embeddings and labels
    num_samples = 100
    embeddings = np.random.randn(num_samples, 768).astype('float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # L2 normalize
    
    labels = np.random.randint(0, 2, num_samples).tolist()
    metadata = [{'id': i, 'dataset': 'dummy'} for i in range(num_samples)]
    
    # Add to index
    print(f"Adding {num_samples} embeddings to index...")
    faiss_index.add(embeddings, labels, metadata)
    
    # Query
    print("Querying for similar cases...")
    query_embedding = np.random.randn(1, 768).astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    results = faiss_index.search_with_metadata(query_embedding, k=5)
    
    print(f"✓ Found {len(results[0])} similar cases:")
    for i, case in enumerate(results[0]):
        print(f"  {i+1}. Distance: {case['distance']:.4f}, Label: {case['label']}")


def example_4_attention_visualization():
    """Example 4: Attention rollout visualization"""
    print("\n" + "="*50)
    print("Example 4: Attention Visualization")
    print("="*50)
    
    # Load model
    model = SwinEmbedder()
    model.eval()
    
    processor = AutoImageProcessor.from_pretrained(
        Config.MODEL_NAME,
        cache_dir=str(Config.MODEL_DIR)
    )
    
    # Create dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    # Process image
    inputs = processor(images=dummy_image, return_tensors="pt")
    
    # Get attention rollout
    print("Computing attention rollout...")
    try:
        attention_map = model.get_attention_rollout(inputs['pixel_values'])
        
        if attention_map is not None:
            print(f"✓ Attention map shape: {attention_map.shape}")
            print(f"✓ Attention map range: [{attention_map.min():.4f}, {attention_map.max():.4f}]")
            
            # Note: In a real scenario, you would visualize this with matplotlib
            # plt.imshow(attention_map, cmap='jet')
            # plt.colorbar()
            # plt.title('Attention Rollout')
            # plt.show()
        else:
            print("⚠ Attention map generation returned None")
    except Exception as e:
        print(f"⚠ Error generating attention map: {e}")


def example_5_api_usage():
    """Example 5: Using the FastAPI endpoints"""
    print("\n" + "="*50)
    print("Example 5: API Usage")
    print("="*50)
    
    api_url = f"http://localhost:{Config.PORT}"
    
    print(f"API should be running at: {api_url}")
    print("\nExample API calls:")
    print("\n1. Health check:")
    print(f"   curl {api_url}/")
    
    print("\n2. Generate embedding:")
    print(f'   curl -X POST "{api_url}/embed" \\')
    print(f'     -H "Content-Type: multipart/form-data" \\')
    print(f'     -F "file=@image.png"')
    
    print("\n3. Retrieve similar cases:")
    print(f'   curl -X POST "{api_url}/retrieve?top_k=5" \\')
    print(f'     -H "Content-Type: multipart/form-data" \\')
    print(f'     -F "file=@image.png"')
    
    print("\n4. Visualize attention:")
    print(f'   curl -X POST "{api_url}/visualize-attention" \\')
    print(f'     -H "Content-Type: multipart/form-data" \\')
    print(f'     -F "file=@image.png"')
    
    print("\n5. Get API info:")
    print(f"   curl {api_url}/info")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("BRAIN - Breast Research Augmented Intelligence Network")
    print("Example Usage Scripts")
    print("="*70)
    
    # Create directories
    Config.create_directories()
    
    try:
        example_1_generate_embedding()
    except Exception as e:
        print(f"✗ Example 1 failed: {e}")
    
    try:
        example_2_batch_embeddings()
    except Exception as e:
        print(f"✗ Example 2 failed: {e}")
    
    try:
        example_3_faiss_retrieval()
    except Exception as e:
        print(f"✗ Example 3 failed: {e}")
    
    try:
        example_4_attention_visualization()
    except Exception as e:
        print(f"✗ Example 4 failed: {e}")
    
    example_5_api_usage()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
    print("\nNext steps:")
    print("1. Download CBIS-DDSM and BreakHis datasets")
    print("2. Run training: python train.py")
    print("3. Start API: python api.py")
    print("4. See README.md for more details")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
