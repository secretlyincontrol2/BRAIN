"""
Configuration for BRAIN - Breast Research Augmented Intelligence Network
Swin Transformer Embedding System with Triplet Loss and FAISS Retrieval
"""
import os
from pathlib import Path


class Config:
    """Configuration class for the embedding system"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
    FAISS_INDEX_DIR = BASE_DIR / "faiss_indices"
    
    # Dataset paths
    CBIS_DDSM_DIR = DATA_DIR / "cbis_ddsm"
    BREAKHIS_DIR = DATA_DIR / "breakhis"
    
    # Model configuration
    MODEL_NAME = "microsoft/swin-base-patch4-window7-224"
    EMBEDDING_DIM = 768  # Default for swin-base
    
    # Training configuration
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Triplet loss configuration
    TRIPLET_MARGIN = 0.5
    TRIPLET_MINING = "hard"  # Options: "hard", "semi-hard", "all"
    
    # Data augmentation
    IMAGE_SIZE = 224
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # FAISS configuration
    FAISS_INDEX_TYPE = "IVF"  # Options: "Flat", "IVF", "HNSW"
    FAISS_NLIST = 100  # Number of clusters for IVF
    FAISS_NPROBE = 10  # Number of clusters to search
    TOP_K_RETRIEVALS = 5  # Number of similar cases to retrieve
    
    # API configuration
    API_TITLE = "BRAIN - Breast Research Augmented Intelligence Network"
    API_DESCRIPTION = """
    Swin Transformer-based embedding system for Retrieval Augmented Generation (RAG)
    
    Features:
    - Domain-specific training on CBIS-DDSM and BreakHis datasets
    - Triplet loss for discriminative metric learning
    - FAISS-powered similarity search
    - Attention rollout visualization for explainability
    - Real-time similar case retrieval
    """
    API_VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Processing configuration
    DEVICE = "cuda"  # Will fallback to cpu if cuda not available
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    SUPPORTED_FORMATS = ["image/jpeg", "image/png", "image/jpg", "image/tiff"]
    
    # Explainability configuration
    ATTENTION_ROLLOUT_ENABLED = True
    ATTENTION_HEAD_FUSION = "mean"  # Options: "mean", "max", "min"
    ATTENTION_DISCARD_RATIO = 0.1
    
    # Checkpoint configuration
    SAVE_CHECKPOINT_EVERY = 5  # epochs
    KEEP_LAST_N_CHECKPOINTS = 3
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.DATA_DIR, cls.MODEL_DIR, cls.LOGS_DIR, 
                         cls.CHECKPOINTS_DIR, cls.FAISS_INDEX_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
