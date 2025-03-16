"""Model trainer for training AI models."""
import logging
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

from ..core.base.base_model import BaseAIModel
from ..core.config.model_config import ModelConfig, TrainingConfig
from ..core.exceptions.base_exceptions import ProcessingError
from .model_registry import ModelRegistry


class ModelTrainer:
    """Trainer for AI models."""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the model trainer.
        
        Args:
            model_registry: Model registry instance.
            device: Device to train on.
        """
        self.logger = logging.getLogger(__name__)
        self.model_registry = model_registry
        self.device = device
        
    async def train_model(
        self,
        model_type: str,
        version: str,
        training_config: TrainingConfig,
        train_data: List[Dict[str, Any]],
        val_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Train a model.
        
        Args:
            model_type: Type of model to train.
            version: Model version.
            training_config: Training configuration.
            train_data: Training data.
            val_data: Optional validation data.
            
        Returns:
            Dictionary with training results.
            
        Raises:
            ProcessingError: If training fails.
        """
        try:
            # Get model
            model = self.model_registry.get_model(model_type, version)
            model.to(self.device)
            
            # Set up optimizer
            optimizer = Adam(
                model.parameters(),
                lr=training_config.learning_rate
            )
            
            # Set up loss function
            criterion = MSELoss()
            
            # Create data loaders
            train_loader = DataLoader(
                train_data,
                batch_size=training_config.batch_size,
                shuffle=True
            )
            
            if val_data:
                val_loader = DataLoader(
                    val_data,
                    batch_size=training_config.batch_size,
                    shuffle=False
                )
                
            # Training loop
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            
            for epoch in range(training_config.num_epochs):
                # Training
                model.train()
                total_train_loss = 0
                
                for batch in train_loader:
                    # Move batch to device
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor)
                        else v
                        for k, v in batch.items()
                    }
                    
                    # Forward pass
                    optimizer.zero_grad()
                    output = model(batch)
                    loss = criterion(output, batch["target"])
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    
                avg_train_loss = total_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                
                # Validation
                if val_data:
                    model.eval()
                    total_val_loss = 0
                    
                    with torch.no_grad():
                        for batch in val_loader:
                            # Move batch to device
                            batch = {
                                k: v.to(self.device) if isinstance(v, torch.Tensor)
                                else v
                                for k, v in batch.items()
                            }
                            
                            # Forward pass
                            output = model(batch)
                            loss = criterion(output, batch["target"])
                            total_val_loss += loss.item()
                            
                    avg_val_loss = total_val_loss / len(val_loader)
                    val_losses.append(avg_val_loss)
                    
                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        self.model_registry.save_model(model, model_type, version)
                        
                # Log progress
                self.logger.info(
                    f"Epoch {epoch + 1}/{training_config.num_epochs}"
                )
                self.logger.info(f"  Train loss: {avg_train_loss:.4f}")
                if val_data:
                    self.logger.info(f"  Val loss: {avg_val_loss:.4f}")
                    
            return {
                "num_epochs": training_config.num_epochs,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss
            }
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise ProcessingError(
                f"Failed to train model: {str(e)}"
            )
            
    async def evaluate_model(
        self,
        model_type: str,
        version: str,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a model.
        
        Args:
            model_type: Type of model to evaluate.
            version: Model version.
            test_data: Test data.
            
        Returns:
            Dictionary with evaluation results.
            
        Raises:
            ProcessingError: If evaluation fails.
        """
        try:
            # Get model
            model = self.model_registry.get_model(model_type, version)
            model.to(self.device)
            model.eval()
            
            # Set up loss function
            criterion = MSELoss()
            
            # Create data loader
            test_loader = DataLoader(
                test_data,
                batch_size=32,  # Fixed batch size for evaluation
                shuffle=False
            )
            
            # Evaluation loop
            total_loss = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    # Move batch to device
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor)
                        else v
                        for k, v in batch.items()
                    }
                    
                    # Forward pass
                    output = model(batch)
                    loss = criterion(output, batch["target"])
                    
                    total_loss += loss.item() * len(batch["target"])
                    total_samples += len(batch["target"])
                    
            avg_loss = total_loss / total_samples
            
            return {
                "num_samples": total_samples,
                "loss": avg_loss
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation error: {str(e)}")
            raise ProcessingError(
                f"Failed to evaluate model: {str(e)}"
            ) 