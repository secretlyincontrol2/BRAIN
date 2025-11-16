"""
FAISS-based similarity search for retrieval-augmented verification
"""
import faiss
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pickle
import logging

from config import Config

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    FAISS index for efficient similarity search in the embedding space.
    
    This implements the retrieval component of the RAG system, enabling
    fast nearest neighbor search to find similar cases for verification.
    """
    
    def __init__(self, embedding_dim: int = None, index_type: str = None):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('Flat', 'IVF', 'HNSW')
        """
        self.embedding_dim = embedding_dim or Config.EMBEDDING_DIM
        self.index_type = index_type or Config.FAISS_INDEX_TYPE
        self.index = None
        self.metadata = []  # Store metadata for each indexed embedding
        self.labels = []  # Store labels for each indexed embedding
        
        self._create_index()
        
        logger.info(f"Initialized FAISS index: type={self.index_type}, dim={self.embedding_dim}")
    
    def _create_index(self):
        """Create FAISS index based on configuration"""
        if self.index_type == "Flat":
            # Exact search using L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
        elif self.index_type == "IVF":
            # Inverted file index for faster approximate search
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.embedding_dim, 
                Config.FAISS_NLIST
            )
            self.requires_training = True
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graph
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def train(self, embeddings: np.ndarray):
        """
        Train the index (required for IVF indices).
        
        Args:
            embeddings: Training embeddings [n_samples, embedding_dim]
        """
        if hasattr(self, 'requires_training') and self.requires_training:
            logger.info(f"Training FAISS index with {len(embeddings)} embeddings")
            self.index.train(embeddings)
            self.requires_training = False
    
    def add(self, embeddings: np.ndarray, labels: List[int], metadata: List[Dict]):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Embeddings to add [n_samples, embedding_dim]
            labels: Corresponding labels
            metadata: Corresponding metadata dictionaries
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} vs {self.embedding_dim}")
        
        # Ensure embeddings are contiguous and float32
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))
        
        # Train if needed
        if hasattr(self, 'requires_training') and self.requires_training:
            self.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata and labels
        self.labels.extend(labels)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} embeddings to index. Total: {self.index.ntotal}")
    
    def search(self, query_embeddings: np.ndarray, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_embeddings: Query embeddings [n_queries, embedding_dim]
            k: Number of neighbors to retrieve
            
        Returns:
            distances: Distances to neighbors [n_queries, k]
            indices: Indices of neighbors [n_queries, k]
        """
        if k is None:
            k = Config.TOP_K_RETRIEVALS
        
        # Ensure embeddings are contiguous and float32
        query_embeddings = np.ascontiguousarray(query_embeddings.astype('float32'))
        
        # Set nprobe for IVF indices
        if self.index_type == "IVF":
            self.index.nprobe = Config.FAISS_NPROBE
        
        # Search
        distances, indices = self.index.search(query_embeddings, k)
        
        return distances, indices
    
    def search_with_metadata(self, query_embeddings: np.ndarray, k: int = None) -> List[List[Dict]]:
        """
        Search and return results with metadata.
        
        Args:
            query_embeddings: Query embeddings [n_queries, embedding_dim]
            k: Number of neighbors to retrieve
            
        Returns:
            List of result lists, each containing dictionaries with:
                - distance: Distance to query
                - label: Label of the retrieved item
                - metadata: Metadata dictionary
        """
        distances, indices = self.search(query_embeddings, k)
        
        results = []
        for i in range(len(query_embeddings)):
            query_results = []
            for j in range(k):
                idx = indices[i, j]
                if idx >= 0 and idx < len(self.metadata):
                    query_results.append({
                        'distance': float(distances[i, j]),
                        'label': self.labels[idx],
                        'metadata': self.metadata[idx]
                    })
            results.append(query_results)
        
        return results
    
    def save(self, path: Path):
        """
        Save the index and metadata to disk.
        
        Args:
            path: Path to save the index
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = str(path.with_suffix('.index'))
        faiss.write_index(self.index, index_path)
        
        # Save metadata and labels
        metadata_path = str(path.with_suffix('.pkl'))
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'labels': self.labels,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }, f)
        
        logger.info(f"Saved FAISS index to {index_path}")
    
    def load(self, path: Path):
        """
        Load the index and metadata from disk.
        
        Args:
            path: Path to load the index from
        """
        path = Path(path)
        
        # Load FAISS index
        index_path = str(path.with_suffix('.index'))
        self.index = faiss.read_index(index_path)
        
        # Load metadata and labels
        metadata_path = str(path.with_suffix('.pkl'))
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.labels = data['labels']
            self.embedding_dim = data['embedding_dim']
            self.index_type = data['index_type']
        
        logger.info(f"Loaded FAISS index from {index_path} with {self.index.ntotal} embeddings")
    
    def remove(self, indices: List[int]):
        """
        Remove embeddings from the index.
        
        Note: Not all FAISS indices support removal.
        """
        if hasattr(self.index, 'remove_ids'):
            id_selector = faiss.IDSelectorBatch(indices)
            self.index.remove_ids(id_selector)
            
            # Remove from metadata and labels
            # This is inefficient for large indices
            indices_set = set(indices)
            self.metadata = [m for i, m in enumerate(self.metadata) if i not in indices_set]
            self.labels = [l for i, l in enumerate(self.labels) if i not in indices_set]
        else:
            raise NotImplementedError(f"Index type {self.index_type} does not support removal")
    
    def __len__(self):
        """Return the number of embeddings in the index"""
        return self.index.ntotal if self.index else 0


def build_index_from_dataloader(model: torch.nn.Module, 
                                dataloader: torch.utils.data.DataLoader,
                                device: str = 'cuda') -> FAISSIndex:
    """
    Build a FAISS index from a dataloader.
    
    Args:
        model: Embedding model
        dataloader: DataLoader with images and labels
        device: Device to run model on
        
    Returns:
        Populated FAISS index
    """
    model.eval()
    model.to(device)
    
    faiss_index = FAISSIndex()
    
    all_embeddings = []
    all_labels = []
    all_metadata = []
    
    logger.info("Building FAISS index from dataloader...")
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label']
            metadata = batch['metadata']
            
            # Extract embeddings
            embeddings = model.extract_embeddings(images)
            
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            labels_np = labels.numpy().tolist()
            
            all_embeddings.append(embeddings_np)
            all_labels.extend(labels_np)
            all_metadata.extend(metadata)
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    # Add to index
    faiss_index.add(all_embeddings, all_labels, all_metadata)
    
    logger.info(f"Built FAISS index with {len(faiss_index)} embeddings")
    
    return faiss_index
