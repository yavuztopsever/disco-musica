"""MIDI model for music generation."""
import logging
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from midiutil import MIDIFile

from ...core.base.base_model import BaseAIModel
from ...core.config.model_config import ModelConfig, TrainingConfig
from ...core.exceptions.base_exceptions import ProcessingError


class MIDIModel(BaseAIModel):
    """MIDI model for music generation."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the MIDI model.
        
        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Model architecture
        self.encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        
        # Move model to device
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        # Encode
        x = self.encoder(x)
        
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
            # Get input MIDI
            midi = inputs.get("midi")
            if midi is None:
                raise ProcessingError("No MIDI input provided")
                
            # Convert to tensor if needed
            if isinstance(midi, (np.ndarray, list)):
                midi = torch.tensor(midi, dtype=torch.float32)
                
            # Add batch dimension if needed
            if midi.dim() == 1:
                midi = midi.unsqueeze(0)
                
            # Move to device
            midi = midi.to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self(midi)
                
            # Convert to numpy array
            output = output.cpu().numpy()
            
            # Convert to MIDI file
            midi_file = self._array_to_midi(output[0])
            
            return {
                "midi": midi_file
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
                    # Get input MIDI
                    midi = batch.get("midi")
                    if midi is None:
                        continue
                        
                    # Convert to tensor if needed
                    if isinstance(midi, (np.ndarray, list)):
                        midi = torch.tensor(midi, dtype=torch.float32)
                        
                    # Add batch dimension if needed
                    if midi.dim() == 1:
                        midi = midi.unsqueeze(0)
                        
                    # Move to device
                    midi = midi.to(self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    output = self(midi)
                    loss = criterion(output, midi)
                    
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
                            # Get input MIDI
                            midi = batch.get("midi")
                            if midi is None:
                                continue
                                
                            # Convert to tensor if needed
                            if isinstance(midi, (np.ndarray, list)):
                                midi = torch.tensor(midi, dtype=torch.float32)
                                
                            # Add batch dimension if needed
                            if midi.dim() == 1:
                                midi = midi.unsqueeze(0)
                                
                            # Move to device
                            midi = midi.to(self.device)
                            
                            # Forward pass
                            output = self(midi)
                            loss = criterion(output, midi)
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
            self.load_state_dict(torch.load(path))
        except Exception as e:
            self.logger.error(f"Load error: {str(e)}")
            raise ProcessingError(
                f"Failed to load model: {str(e)}"
            )
            
    def _array_to_midi(self, array: np.ndarray) -> bytes:
        """Convert numpy array to MIDI file.
        
        Args:
            array: Numpy array of MIDI data.
            
        Returns:
            MIDI file as bytes.
        """
        try:
            # Create MIDI file
            midi = MIDIFile(1)
            
            # Add notes
            for i in range(len(array)):
                if array[i] > 0.5:  # Threshold for note presence
                    midi.addNote(
                        track=0,
                        channel=0,
                        pitch=i,
                        time=i/4,  # Assuming 4/4 time
                        duration=0.25,
                        volume=100
                    )
                    
            # Write to bytes
            import io
            output = io.BytesIO()
            midi.writeFile(output)
            return output.getvalue()
            
        except Exception as e:
            self.logger.error(f"MIDI conversion error: {str(e)}")
            raise ProcessingError(
                f"Failed to convert array to MIDI: {str(e)}"
            ) 