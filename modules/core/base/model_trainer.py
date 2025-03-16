"""Framework for training models."""
from typing import Dict, Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base.base_model import BaseAIModel
from ..config.model_config import TrainingConfig


class ModelTrainer:
    """Framework for training models."""
    
    def __init__(
        self, 
        model: BaseAIModel,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer_fn: Callable = None,
        scheduler_fn: Callable = None,
        device: str = None
    ):
        """Initialize the trainer.
        
        Args:
            model: Model to train.
            config: Training configuration.
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
            optimizer_fn: Function to create optimizer.
            scheduler_fn: Function to create learning rate scheduler.
            device: Device to train on (auto-detected if None).
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Set up optimizer
        if optimizer_fn is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer_fn(model.parameters())
            
        # Set up scheduler
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer)
        else:
            self.scheduler = None
            
        # Initialize trackers
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self) -> float:
        """Train one epoch.
        
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = outputs["loss"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_val
                )
                
            self.optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
        
    def validate(self) -> float:
        """Run validation.
        
        Returns:
            Average validation loss.
        """
        if not self.val_loader:
            return 0.0
            
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                         for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                loss = outputs["loss"]
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        # Check for best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            return avg_loss
            
    def train(self, num_epochs: int) -> Dict[str, Any]:
        """Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train.
            
        Returns:
            Dictionary with training results.
        """
        for epoch in range(num_epochs):
            self.current_epoch += 1
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Log progress
            print(f"Epoch {self.current_epoch}/{num_epochs}")
            print(f"  Train loss: {train_loss:.4f}")
            if val_loss:
                print(f"  Val loss: {val_loss:.4f}")
                
        return {
            "num_epochs": num_epochs,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        } 