"""
Swin Transformer Embedding Model with Triplet Loss
"""
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SwinModel
from typing import Optional, Tuple
import numpy as np
import logging

from config import Config

logger = logging.getLogger(__name__)


class SwinEmbedder(nn.Module):
    """
    Swin Transformer-based embedder for breast imaging.
    
    This model uses a pre-trained Swin Transformer as a backbone
    and is fine-tuned using triplet loss to learn a discriminative
    embedding space where Euclidean distance corresponds to semantic similarity.
    """
    
    def __init__(self, model_name: Optional[str] = None, 
                 embedding_dim: Optional[int] = None,
                 pretrained: bool = True):
        """
        Initialize the Swin Embedder.
        
        Args:
            model_name: HuggingFace model name
            embedding_dim: Dimension of output embeddings
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.model_name = model_name or Config.MODEL_NAME
        self.embedding_dim = embedding_dim or Config.EMBEDDING_DIM
        
        logger.info(f"Initializing SwinEmbedder with {self.model_name}")
        
        # Load Swin Transformer backbone
        self.backbone = SwinModel.from_pretrained(
            self.model_name,
            cache_dir=str(Config.MODEL_DIR)
        ) if pretrained else SwinModel.from_config(self.model_name)
        
        # Get the hidden size from the backbone
        self.hidden_size = self.backbone.config.hidden_size
        
        # Projection head (optional, for dimension reduction or expansion)
        if self.embedding_dim != self.hidden_size:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, self.embedding_dim),
                nn.BatchNorm1d(self.embedding_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.projection = nn.Identity()
        
        # L2 normalization for embeddings (important for metric learning)
        self.normalize = True
        
        logger.info(f"Model initialized. Hidden size: {self.hidden_size}, Embedding dim: {self.embedding_dim}")
    
    def forward(self, pixel_values: torch.Tensor, 
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            pixel_values: Input images [batch_size, 3, height, width]
            return_attention: Whether to return attention weights for visualization
            
        Returns:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            attentions: Optional attention weights for explainability
        """
        # Get features from backbone
        outputs = self.backbone(
            pixel_values,
            output_attentions=return_attention,
            output_hidden_states=False
        )
        
        # Use pooler output (global average pooled features)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # Project to embedding dimension
        embeddings = self.projection(pooled_output)  # [batch_size, embedding_dim]
        
        # L2 normalization
        if self.normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        attentions = outputs.attentions if return_attention else None
        
        return embeddings, attentions
    
    def extract_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings without gradient computation.
        
        Args:
            pixel_values: Input images
            
        Returns:
            embeddings: L2-normalized embeddings
        """
        with torch.no_grad():
            embeddings, _ = self.forward(pixel_values)
        return embeddings
    
    def get_attention_rollout(self, pixel_values: torch.Tensor) -> np.ndarray:
        """
        Compute attention rollout for explainability.
        
        Attention rollout aggregates attention from all layers to show
        which image regions the model focuses on.
        
        Args:
            pixel_values: Input image [1, 3, height, width]
            
        Returns:
            attention_map: Attention rollout map [height, width]
        """
        with torch.no_grad():
            _, attentions = self.forward(pixel_values, return_attention=True)
        
        if attentions is None:
            return None
        
        # attentions is a tuple of attention tensors, one per layer
        # Each attention tensor has shape [batch_size, num_heads, seq_len, seq_len]
        
        # Average across heads
        attention_matrices = []
        for attention in attentions:
            # Average over heads: [batch_size, seq_len, seq_len]
            if Config.ATTENTION_HEAD_FUSION == "mean":
                attention = attention.mean(dim=1)
            elif Config.ATTENTION_HEAD_FUSION == "max":
                attention = attention.max(dim=1)[0]
            elif Config.ATTENTION_HEAD_FUSION == "min":
                attention = attention.min(dim=1)[0]
            
            attention_matrices.append(attention)
        
        # Perform attention rollout
        rollout = attention_matrices[0]
        for attention in attention_matrices[1:]:
            rollout = torch.matmul(attention, rollout)
        
        # Get attention for CLS token or use mean
        # Shape: [batch_size, seq_len]
        rollout = rollout[:, 0, 1:]  # First token attending to all others
        
        # Reshape to spatial dimensions
        # For Swin Transformer with 224x224 input and patch size 4
        # We have 56x56 = 3136 patches initially
        seq_len = rollout.shape[1]
        grid_size = int(np.sqrt(seq_len))
        
        attention_map = rollout.reshape(1, grid_size, grid_size)
        attention_map = attention_map.cpu().numpy()[0]
        
        return attention_map


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.
    
    The loss encourages embeddings of the same class (anchor-positive)
    to be closer than embeddings of different classes (anchor-negative)
    by at least a margin.
    
    Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
    """
    
    def __init__(self, margin: float = None, mining: str = None):
        """
        Args:
            margin: Margin for triplet loss
            mining: Type of triplet mining ('hard', 'semi-hard', 'all')
        """
        super().__init__()
        self.margin = margin or Config.TRIPLET_MARGIN
        self.mining = mining or Config.TRIPLET_MINING
        
        logger.info(f"Initialized TripletLoss with margin={self.margin}, mining={self.mining}")
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            embeddings: Batch of embeddings [batch_size, embedding_dim]
            labels: Batch of labels [batch_size]
            
        Returns:
            loss: Triplet loss value
        """
        # Compute pairwise distances
        distances = self._pairwise_distances(embeddings)
        
        # Mine triplets
        if self.mining == "hard":
            loss = self._hard_triplet_loss(distances, labels)
        elif self.mining == "semi-hard":
            loss = self._semi_hard_triplet_loss(distances, labels)
        else:  # all
            loss = self._batch_all_triplet_loss(distances, labels)
        
        return loss
    
    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances"""
        # embeddings: [batch_size, embedding_dim]
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        
        # distances[i, j] = ||embeddings[i] - embeddings[j]||^2
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.clamp(distances, min=0.0)
        
        # Fix numerical errors
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
        
        return distances
    
    def _get_triplet_mask(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get masks for valid positive and negative pairs"""
        # Positive mask: same label, different sample
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        positive_mask = labels_equal & ~indices_equal
        
        # Negative mask: different label
        negative_mask = ~labels_equal
        
        return positive_mask, negative_mask
    
    def _hard_triplet_loss(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Hard triplet mining: for each anchor, use the hardest positive and negative
        """
        positive_mask, negative_mask = self._get_triplet_mask(labels)
        
        # For each anchor, find the hardest positive (furthest positive)
        masked_pos_distances = distances * positive_mask.float()
        hardest_positive = masked_pos_distances.max(dim=1)[0]
        
        # For each anchor, find the hardest negative (closest negative)
        masked_neg_distances = distances.clone()
        masked_neg_distances[~negative_mask] = float('inf')
        hardest_negative = masked_neg_distances.min(dim=1)[0]
        
        # Compute triplet loss
        loss = torch.clamp(hardest_positive - hardest_negative + self.margin, min=0.0)
        loss = loss.mean()
        
        return loss
    
    def _semi_hard_triplet_loss(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Semi-hard triplet mining: negatives that are further than positive but within margin
        """
        positive_mask, negative_mask = self._get_triplet_mask(labels)
        
        batch_size = labels.size(0)
        total_loss = 0.0
        num_triplets = 0
        
        for i in range(batch_size):
            # Get distances for anchor i
            pos_distances = distances[i][positive_mask[i]]
            neg_distances = distances[i][negative_mask[i]]
            
            if len(pos_distances) == 0 or len(neg_distances) == 0:
                continue
            
            # For each positive, find semi-hard negatives
            for pos_dist in pos_distances:
                # Semi-hard: d(a,n) > d(a,p) but d(a,n) < d(a,p) + margin
                semi_hard_negatives = neg_distances[
                    (neg_distances > pos_dist) & (neg_distances < pos_dist + self.margin)
                ]
                
                if len(semi_hard_negatives) > 0:
                    # Use the hardest semi-hard negative
                    neg_dist = semi_hard_negatives.min()
                    loss = pos_dist - neg_dist + self.margin
                    total_loss += torch.clamp(loss, min=0.0)
                    num_triplets += 1
        
        if num_triplets > 0:
            return total_loss / num_triplets
        else:
            return torch.tensor(0.0, device=distances.device)
    
    def _batch_all_triplet_loss(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Batch-all triplet loss: compute loss for all valid triplets
        """
        positive_mask, negative_mask = self._get_triplet_mask(labels)
        
        batch_size = labels.size(0)
        
        # For each anchor, create triplets with all valid positives and negatives
        # distances: [batch_size, batch_size]
        # Expand dimensions for broadcasting
        anchor_positive_dist = distances.unsqueeze(2)  # [batch_size, batch_size, 1]
        anchor_negative_dist = distances.unsqueeze(1)  # [batch_size, 1, batch_size]
        
        # Compute triplet loss for all combinations
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        
        # Create mask for valid triplets
        # Valid if: anchor != positive, anchor != negative, positive != negative, 
        #           anchor and positive have same label, anchor and negative have different labels
        positive_mask_3d = positive_mask.unsqueeze(2)  # [batch_size, batch_size, 1]
        negative_mask_3d = negative_mask.unsqueeze(1)  # [batch_size, 1, batch_size]
        valid_triplets = positive_mask_3d & negative_mask_3d
        
        # Apply mask
        triplet_loss = triplet_loss * valid_triplets.float()
        triplet_loss = torch.clamp(triplet_loss, min=0.0)
        
        # Average over valid triplets
        num_valid_triplets = valid_triplets.sum().float()
        if num_valid_triplets > 0:
            triplet_loss = triplet_loss.sum() / num_valid_triplets
        else:
            triplet_loss = torch.tensor(0.0, device=distances.device)
        
        return triplet_loss
