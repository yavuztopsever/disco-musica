"""Text model for music generation."""
import logging
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel

from ...core.base.base_model import BaseAIModel
from ...core.config.model_config import ModelConfig, TrainingConfig
from ...core.exceptions.base_exceptions import ProcessingError


class TextModel(BaseAIModel):
    """Text model for music generation."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the text model.
        
        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )
        self.base_model = AutoModel.from_pretrained(
            "bert-base-uncased"
        )
        
        # Model architecture
        self.encoder = nn.Sequential(
            nn.Linear(768, 512),  # BERT hidden size is 768
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        
        # Move models to device
        self.base_model.to(self.device)
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.base_model(x)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
            
        # Encode
        x = self.encoder(embeddings)
        
        # Decode
        x = self.decoder(x)
        
        return x
        
    async def predict(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run inference on the model.
        
        Args:
            inputs: Dictionary of input tensors or values.
            
        Returns:
            Dictionary of output tensors or values.
        """
        try:
            # Get input text
            text = inputs.get("text")
            if text is None:
                raise ProcessingError("No text input provided")
                
            # Tokenize text
            tokens = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            
            # Run inference
            with torch.no_grad():
                output = self(tokens["input_ids"])
                
            # Convert to numpy array
            output = output.cpu().numpy()
            
            return {
                "embedding": output[0]  # Return first (and only) embedding
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise ProcessingError(
                f"Failed to run prediction: {str(e)}"
            )
            
    async def train(
        self,
        training_config: TrainingConfig,
        train_data: List[Dict[str, Any]],
        val_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            training_config: Training configuration.
            train_data: Training data.
            val_data: Optional validation data.
            
        Returns:
            Dictionary with training results.
        """
        try:
            # Set up optimizer
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=training_config.learning_rate
            )
            
            # Set up loss function
            criterion = nn.MSELoss()
            
            # Training loop
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            
            for epoch in range(training_config.num_epochs):
                # Training
                self.train()
                total_train_loss = 0
                
                for batch in train_data:
                    # Get input text
                    text = batch.get("text")
                    if text is None:
                        continue
                        
                    # Tokenize text
                    tokens = self.tokenizer(
                        text,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    # Move to device
                    tokens = {k: v.to(self.device) for k, v in tokens.items()}
                    
                    # Forward pass
                    optimizer.zero_grad()
                    output = self(tokens["input_ids"])
                    loss = criterion(output, batch["target"])
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    
                avg_train_loss = total_train_loss / len(train_data)
                train_losses.append(avg_train_loss)
                
                # Validation
                if val_data:
                    self.eval()
                    total_val_loss = 0
                    
                    with torch.no_grad():
                        for batch in val_data:
                            # Get input text
                            text = batch.get("text")
                            if text is None:
                                continue
                                
                            # Tokenize text
                            tokens = self.tokenizer(
                                text,
                                padding=True,
                                truncation=True,
                                max_length=512,
                                return_tensors="pt"
                            )
                            
                            # Move to device
                            tokens = {k: v.to(self.device) for k, v in tokens.items()}
                            
                            # Forward pass
                            output = self(tokens["input_ids"])
                            loss = criterion(output, batch["target"])
                            total_val_loss += loss.item()
                            
                    avg_val_loss = total_val_loss / len(val_data)
                    val_losses.append(avg_val_loss)
                    
                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        self.save(self.config.weights_path)
                        
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
            
    def save(self, path: str) -> None:
        """Save model weights to path.
        
        Args:
            path: Path to save model weights.
        """
        try:
            # Save base model
            self.base_model.save_pretrained(f"{path}_base")
            
            # Save tokenizer
            self.tokenizer.save_pretrained(f"{path}_base")
            
            # Save custom layers
            torch.save(self.state_dict(), path)
        except Exception as e:
            self.logger.error(f"Save error: {str(e)}")
            raise ProcessingError(
                f"Failed to save model: {str(e)}"
            )
            
    def load(self, path: str) -> None:
        """Load model weights from path.
        
        Args:
            path: Path to model weights file.
        """
        try:
            # Load base model
            self.base_model = AutoModel.from_pretrained(f"{path}_base")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(f"{path}_base")
            
            # Load custom layers
            self.load_state_dict(torch.load(path))
            
            # Move models to device
            self.base_model.to(self.device)
            self.to(self.device)
        except Exception as e:
            self.logger.error(f"Load error: {str(e)}")
            raise ProcessingError(
                f"Failed to load model: {str(e)}"
            ) 