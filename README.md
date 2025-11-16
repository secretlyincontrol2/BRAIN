# BRAIN - Breast Research Augmented Intelligence Network

A Swin Transformer-based embedding system for Retrieval Augmented Generation (RAG) in breast cancer imaging, featuring triplet loss training, FAISS-powered similarity search, and attention-based explainability.

## Overview

BRAIN implements a complete pipeline for medical image analysis with the following key components:

1. **Domain-Specific Data Curation**: Support for CBIS-DDSM (mammography) and BreakHis (histopathology) datasets
2. **Hierarchical Feature Extraction**: Swin Transformer backbone for high-dimensional embeddings
3. **Discriminative Metric Learning**: Triplet loss training for learning semantic similarity
4. **Retrieval-Augmented Verification**: FAISS integration for fast large-scale similarity search
5. **Clinical Translation and Explainability**: Attention Rollout visualization and similar case retrieval

## Features

- 🔬 **Multi-Modal Dataset Support**: CBIS-DDSM (mammography) and BreakHis (histopathology)
- 🧠 **Swin Transformer Embeddings**: State-of-the-art vision transformer architecture
- 📐 **Triplet Loss Training**: Learn discriminative embeddings where distance = similarity
- 🚀 **FAISS Similarity Search**: Fast retrieval of similar cases at scale
- 🎯 **Attention Visualization**: Explainable AI through attention rollout
- 🌐 **FastAPI REST API**: Easy integration with web applications
- 📊 **TensorBoard Logging**: Track training metrics in real-time

## Architecture

```
Input Image
    ↓
Swin Transformer (Backbone)
    ↓
Embedding (768-dim)
    ↓
L2 Normalization
    ↓
├─→ Triplet Loss (Training)
└─→ FAISS Index (Retrieval)
        ↓
    Similar Cases (RAG)
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/secretlyincontrol2/BRAIN.git
cd BRAIN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

### CBIS-DDSM (Mammography)

1. Download from: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
2. Extract to: `data/cbis_ddsm/`
3. Expected structure:
```
data/cbis_ddsm/
├── Mass-Training/
├── Mass-Test/
├── Calc-Training/
└── Calc-Test/
```

### BreakHis (Histopathology)

1. Download from: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
2. Extract to: `data/breakhis/`
3. Expected structure:
```
data/breakhis/
└── BreaKHis_v1/
    └── histology_slides/
        └── breast/
            ├── benign/
            └── malignant/
```

## Training

Train the Swin Transformer embedder with triplet loss:

```bash
# Basic training
python train.py

# Custom parameters
python train.py --epochs 50 --batch_size 32 --lr 1e-4

# Resume from checkpoint
python train.py --resume checkpoints/best_model.pth
```

Training outputs:
- **Checkpoints**: Saved to `checkpoints/`
- **TensorBoard logs**: Saved to `logs/`
- **FAISS index**: Built automatically after training

### Monitor Training

```bash
tensorboard --logdir logs/
```

## API Usage

### Start the API Server

```bash
# Development mode
python api.py

# Production mode
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/
```

#### 2. Generate Embedding
```bash
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.png"
```

Response:
```json
{
  "embedding": [0.123, -0.456, ...],
  "embedding_dim": 768,
  "model_name": "microsoft/swin-base-patch4-window7-224"
}
```

#### 3. Retrieve Similar Cases (RAG)
```bash
curl -X POST "http://localhost:8000/retrieve?top_k=5" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.png"
```

Response:
```json
{
  "query_embedding": [...],
  "similar_cases": [
    {
      "distance": 0.234,
      "label": 1,
      "label_name": "Malignant",
      "metadata": {
        "dataset": "cbis_ddsm",
        "type": "Mass"
      }
    },
    ...
  ],
  "top_k": 5
}
```

#### 4. Visualize Attention (Explainability)
```bash
curl -X POST "http://localhost:8000/visualize-attention" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.png"
```

Response:
```json
{
  "attention_map_base64": "iVBORw0KGgoAAAANS...",
  "overlay_image_base64": "iVBORw0KGgoAAAANS..."
}
```

## Configuration

Edit `config.py` to customize:

```python
# Model
MODEL_NAME = "microsoft/swin-base-patch4-window7-224"
EMBEDDING_DIM = 768

# Training
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# Triplet Loss
TRIPLET_MARGIN = 0.5
TRIPLET_MINING = "hard"  # Options: "hard", "semi-hard", "all"

# FAISS
FAISS_INDEX_TYPE = "IVF"  # Options: "Flat", "IVF", "HNSW"
TOP_K_RETRIEVALS = 5
```

## Python API

### Generate Embeddings

```python
from PIL import Image
from model import SwinEmbedder
import torch

# Load model
model = SwinEmbedder()
model.eval()

# Load image
image = Image.open('image.png').convert('RGB')

# Process and generate embedding
from transformers import AutoImageProcessor
processor = AutoImageProcessor.from_pretrained('microsoft/swin-base-patch4-window7-224')
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    embedding, _ = model(inputs['pixel_values'])

print(f"Embedding shape: {embedding.shape}")  # [1, 768]
```

### Retrieve Similar Cases

```python
from faiss_retrieval import FAISSIndex

# Load FAISS index
faiss_index = FAISSIndex()
faiss_index.load('faiss_indices/validation_index')

# Search
results = faiss_index.search_with_metadata(embedding.numpy(), k=5)

for i, case in enumerate(results[0]):
    print(f"{i+1}. Distance: {case['distance']:.3f}, Label: {case['label_name']}")
```

### Visualize Attention

```python
import matplotlib.pyplot as plt

# Get attention rollout
attention_map = model.get_attention_rollout(inputs['pixel_values'])

# Visualize
plt.imshow(attention_map, cmap='jet')
plt.colorbar()
plt.title('Attention Rollout')
plt.show()
```

## Project Structure

```
BRAIN/
├── api.py                 # FastAPI application
├── config.py              # Configuration settings
├── datasets.py            # Dataset loaders (CBIS-DDSM, BreakHis)
├── model.py               # Swin Transformer embedder + Triplet Loss
├── faiss_retrieval.py     # FAISS similarity search
├── train.py               # Training script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Datasets (not included)
│   ├── cbis_ddsm/
│   └── breakhis/
├── models/               # Pre-trained model cache
├── checkpoints/          # Training checkpoints
├── logs/                 # TensorBoard logs
└── faiss_indices/        # FAISS indices
```

## Key Technologies

- **Swin Transformer**: Hierarchical vision transformer with shifted windows
- **Triplet Loss**: Metric learning for discriminative embeddings
- **FAISS**: Facebook AI Similarity Search for efficient retrieval
- **FastAPI**: Modern, fast web framework for APIs
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace transformers library

## Citation

If you use BRAIN in your research, please cite:

```bibtex
@software{brain2025,
  title={BRAIN: Breast Research Augmented Intelligence Network},
  author={Timilehin Adedayo, Moyinoluwa Aina},
  year={2025},
  url={https://github.com/secretlyincontrol2/BRAIN}
}
```

## References

- Swin Transformer: [Liu et al., ICCV 2021](https://arxiv.org/abs/2103.14030)
- Triplet Loss: [Schroff et al., CVPR 2015](https://arxiv.org/abs/1503.03832)
- FAISS: [Johnson et al., 2019](https://arxiv.org/abs/1702.08734)
- CBIS-DDSM: [Lee et al., 2017](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
- BreakHis: [Spanhol et al., 2016](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

