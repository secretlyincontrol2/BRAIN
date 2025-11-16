"""
Training script for Swin Transformer embedder with triplet loss
on CBIS-DDSM and BreakHis datasets
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
import argparse

from config import Config
from model import SwinEmbedder, TripletLoss
from datasets import create_dataloaders
from faiss_retrieval import build_index_from_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for Swin Transformer embedder with triplet loss"""
    
    def __init__(self, model: SwinEmbedder, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = None):
        """
        Initialize trainer.
        
        Args:
            model: SwinEmbedder model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or self._get_device()
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function
        self.criterion = TripletLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=Config.NUM_EPOCHS,
            eta_min=1e-6
        )
        
        # Tensorboard writer
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(str(Config.LOGS_DIR / f'run_{timestamp}'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    def _get_device(self):
        """Auto-detect device"""
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{Config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            embeddings, _ = self.model(images)
            
            # Compute loss
            loss = self.criterion(embeddings, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                embeddings, _ = self.model(images)
                
                # Compute loss
                loss = self.criterion(embeddings, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs: int = None):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        if num_epochs is None:
            num_epochs = Config.NUM_EPOCHS
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}")
            
            # Log to tensorboard
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Step scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % Config.SAVE_CHECKPOINT_EVERY == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
        
        logger.info("Training completed!")
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        # Close tensorboard writer
        self.writer.close()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = Config.CHECKPOINTS_DIR / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': {
                'model_name': self.model.model_name,
                'embedding_dim': self.model.embedding_dim,
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Manage checkpoint retention
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints"""
        checkpoint_files = sorted(
            Config.CHECKPOINTS_DIR.glob('checkpoint_epoch_*.pth'),
            key=lambda x: x.stat().st_mtime
        )
        
        # Keep best_model.pth and final_model.pth, only delete checkpoint_epoch_*.pth
        if len(checkpoint_files) > Config.KEEP_LAST_N_CHECKPOINTS:
            for old_checkpoint in checkpoint_files[:-Config.KEEP_LAST_N_CHECKPOINTS]:
                old_checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch + 1}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Swin Transformer embedder')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    args = parser.parse_args()
    
    # Create directories
    Config.create_directories()
    
    # Update config from args
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.lr:
        Config.LEARNING_RATE = args.lr
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataloaders = create_dataloaders(Config.BATCH_SIZE)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    if len(train_loader.dataset) == 0:
        logger.error("No training data found! Please download CBIS-DDSM and BreakHis datasets.")
        logger.error("CBIS-DDSM: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM")
        logger.error("BreakHis: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/")
        return
    
    # Create model
    logger.info("Creating model...")
    model = SwinEmbedder()
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    
    # Train
    trainer.train(args.epochs)
    
    # Build FAISS index from validation set
    logger.info("Building FAISS index from validation set...")
    faiss_index = build_index_from_dataloader(model, val_loader, trainer.device)
    
    # Save FAISS index
    index_path = Config.FAISS_INDEX_DIR / 'validation_index'
    faiss_index.save(index_path)
    logger.info(f"FAISS index saved to {index_path}")


if __name__ == '__main__':
    main()
