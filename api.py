"""
FastAPI application for BRAIN - Breast Research Augmented Intelligence Network
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from PIL import Image
import io
import numpy as np
import logging
from pathlib import Path
import base64
from transformers import AutoImageProcessor
import matplotlib.pyplot as plt
import cv2

from config import Config
from model import SwinEmbedder
from faiss_retrieval import FAISSIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and index
model: Optional[SwinEmbedder] = None
processor: Optional[AutoImageProcessor] = None
faiss_index: Optional[FAISSIndex] = None
device: str = None


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation"""
    embedding: List[float]
    embedding_dim: int
    model_name: str


class SimilarCase(BaseModel):
    """Model for a similar case"""
    distance: float
    label: int
    label_name: str
    metadata: Dict[str, Any]


class RetrievalResponse(BaseModel):
    """Response model for retrieval"""
    query_embedding: List[float]
    similar_cases: List[SimilarCase]
    top_k: int


class AttentionVisualizationResponse(BaseModel):
    """Response model for attention visualization"""
    attention_map_base64: str
    overlay_image_base64: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    faiss_index_loaded: bool
    faiss_index_size: int
    device: str


@app.on_event("startup")
async def startup_event():
    """Initialize model and FAISS index on startup"""
    global model, processor, faiss_index, device
    
    logger.info("Starting up BRAIN API...")
    
    # Create directories
    Config.create_directories()
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model
    try:
        logger.info("Loading Swin Transformer model...")
        model = SwinEmbedder()
        
        # Try to load trained checkpoint
        checkpoint_path = Config.CHECKPOINTS_DIR / 'best_model.pth'
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded trained model checkpoint")
        else:
            logger.warning("No trained checkpoint found. Using pretrained weights.")
        
        model.to(device)
        model.eval()
        
        # Load processor
        processor = AutoImageProcessor.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=str(Config.MODEL_DIR)
        )
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        processor = None
    
    # Load FAISS index
    try:
        index_path = Config.FAISS_INDEX_DIR / 'validation_index'
        if index_path.with_suffix('.index').exists():
            logger.info(f"Loading FAISS index from {index_path}")
            faiss_index = FAISSIndex()
            faiss_index.load(index_path)
            logger.info(f"FAISS index loaded with {len(faiss_index)} embeddings")
        else:
            logger.warning("No FAISS index found. Retrieval will not be available.")
            faiss_index = None
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        faiss_index = None
    
    logger.info("BRAIN API startup complete")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        faiss_index_loaded=faiss_index is not None,
        faiss_index_size=len(faiss_index) if faiss_index else 0,
        device=device
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embedding(file: UploadFile = File(...)):
    """
    Generate embedding for an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        
    Returns:
        Embedding vector and metadata
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file
    if file.content_type not in Config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported: {Config.SUPPORTED_FORMATS}"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        # Generate embedding
        with torch.no_grad():
            embedding, _ = model(pixel_values)
        
        # Convert to list
        embedding_list = embedding.cpu().numpy()[0].tolist()
        
        return EmbeddingResponse(
            embedding=embedding_list,
            embedding_dim=len(embedding_list),
            model_name=Config.MODEL_NAME
        )
    
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_similar_cases(
    file: UploadFile = File(...),
    top_k: int = Query(default=Config.TOP_K_RETRIEVALS, ge=1, le=20)
):
    """
    Retrieve similar cases for an uploaded image.
    
    This implements the retrieval-augmented verification component,
    finding similar cases from the indexed database.
    
    Args:
        file: Image file
        top_k: Number of similar cases to retrieve
        
    Returns:
        Query embedding and list of similar cases
    """
    if model is None or faiss_index is None:
        raise HTTPException(
            status_code=503, 
            detail="Model or FAISS index not loaded"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        # Generate embedding
        with torch.no_grad():
            embedding, _ = model(pixel_values)
        
        # Search in FAISS index
        embedding_np = embedding.cpu().numpy()
        results = faiss_index.search_with_metadata(embedding_np, k=top_k)
        
        # Format results
        similar_cases = []
        for result in results[0]:
            label_name = "Malignant" if result['label'] == 1 else "Benign"
            similar_cases.append(
                SimilarCase(
                    distance=result['distance'],
                    label=result['label'],
                    label_name=label_name,
                    metadata=result['metadata']
                )
            )
        
        return RetrievalResponse(
            query_embedding=embedding.cpu().numpy()[0].tolist(),
            similar_cases=similar_cases,
            top_k=top_k
        )
    
    except Exception as e:
        logger.error(f"Error retrieving similar cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize-attention", response_model=AttentionVisualizationResponse)
async def visualize_attention(file: UploadFile = File(...)):
    """
    Generate attention rollout visualization for explainability.
    
    This provides visual explanation of which regions the model
    focuses on when generating embeddings.
    
    Args:
        file: Image file
        
    Returns:
        Base64-encoded attention map and overlay image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        original_size = image.size
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        # Get attention rollout
        attention_map = model.get_attention_rollout(pixel_values)
        
        if attention_map is None:
            raise HTTPException(status_code=500, detail="Failed to generate attention map")
        
        # Resize attention map to match original image size
        attention_map_resized = cv2.resize(
            attention_map, 
            original_size, 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Normalize to 0-255
        attention_map_normalized = (attention_map_resized - attention_map_resized.min()) / \
                                  (attention_map_resized.max() - attention_map_resized.min())
        attention_map_uint8 = (attention_map_normalized * 255).astype(np.uint8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(attention_map_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        image_np = np.array(image)
        overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
        
        # Convert to base64
        def image_to_base64(img_array):
            img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode()
        
        attention_map_base64 = image_to_base64(heatmap)
        overlay_image_base64 = image_to_base64(overlay)
        
        return AttentionVisualizationResponse(
            attention_map_base64=attention_map_base64,
            overlay_image_base64=overlay_image_base64
        )
    
    except Exception as e:
        logger.error(f"Error visualizing attention: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def get_info():
    """Get API information"""
    return {
        "title": Config.API_TITLE,
        "version": Config.API_VERSION,
        "model": Config.MODEL_NAME,
        "embedding_dim": Config.EMBEDDING_DIM,
        "triplet_margin": Config.TRIPLET_MARGIN,
        "faiss_index_type": Config.FAISS_INDEX_TYPE,
        "datasets": ["CBIS-DDSM", "BreakHis"],
        "endpoints": {
            "/": "Health check",
            "/embed": "Generate embedding for an image",
            "/retrieve": "Retrieve similar cases (RAG)",
            "/visualize-attention": "Generate attention visualization for explainability",
            "/info": "API information"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True
    )
