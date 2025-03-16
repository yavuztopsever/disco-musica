"""Text to Music Model for generating music from text prompts."""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import logging

from .base_model import BaseModel
from ..processors.text_processor import TextProcessor
from ..processors.audio_processor import AudioProcessor
from ..exceptions.base_exceptions import (
    ModelNotFoundError,
    ValidationError,
    ProcessingError
)


class TextToMusicModel(BaseModel):
    """Model for generating music from text prompts.
    
    This model takes text descriptions as input and generates corresponding
    musical audio. It uses a transformer-based architecture to convert text
    embeddings into audio waveforms.
    """
    
    def __init__(
        self,
        version: str,
        config: Dict[str, Any],
        device: Optional[str] = None
    ):
        """Initialize the text to music model.
        
        Args:
            version: Model version.
            config: Model configuration.
            device: Optional device to run model on.
        """
        super().__init__(version, config, device)
        
        # Initialize processors
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor()
        
        # Model architecture parameters
        self.embedding_dim = config.get("embedding_dim", 512)
        self.hidden_dim = config.get("hidden_dim", 1024)
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        
        # Audio parameters
        self.sample_rate = config.get("sample_rate", 44100)
        self.hop_length = config.get("hop_length", 256)
        self.n_mels = config.get("n_mels", 128)
        
        # Build model
        self._build_model()
        
    def _build_model(self) -> None:
        """Build the model architecture."""
        try:
            # Text encoder
            self.text_encoder = nn.ModuleDict({
                "embedding": nn.Embedding(
                    self.config["vocab_size"],
                    self.embedding_dim
                ),
                "transformer": nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.embedding_dim,
                        nhead=self.num_heads,
                        dim_feedforward=self.hidden_dim,
                        dropout=self.dropout,
                        batch_first=True
                    ),
                    num_layers=self.num_layers
                )
            })
            
            # Audio decoder
            self.audio_decoder = nn.ModuleDict({
                "upsampler": nn.Sequential(
                    nn.Linear(self.embedding_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4)
                ),
                "conv_stack": nn.Sequential(
                    nn.ConvTranspose1d(
                        self.hidden_dim * 4,
                        self.hidden_dim * 2,
                        kernel_size=8,
                        stride=4,
                        padding=2
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose1d(
                        self.hidden_dim * 2,
                        self.hidden_dim,
                        kernel_size=8,
                        stride=4,
                        padding=2
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose1d(
                        self.hidden_dim,
                        1,
                        kernel_size=8,
                        stride=4,
                        padding=2
                    ),
                    nn.Tanh()
                )
            })
            
            # Move to device
            self.text_encoder.to(self.device)
            self.audio_decoder.to(self.device)
            
            # Set model reference
            self.model = nn.ModuleDict({
                "encoder": self.text_encoder,
                "decoder": self.audio_decoder
            })
            
        except Exception as e:
            raise ProcessingError(f"Error building model: {e}")
            
    def load(self) -> None:
        """Load model weights and prepare for inference."""
        try:
            # Load weights
            self.load_weights()
            
            # Set to eval mode
            self.model.eval()
            
            # Warm up model
            self.warm_up()
            
            self.is_loaded = True
            
        except Exception as e:
            raise ProcessingError(f"Error loading model: {e}")
            
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate music from text input.
        
        Args:
            inputs: Dictionary containing:
                - text: Input text prompt
                - duration: Optional target duration in seconds
                - temperature: Optional sampling temperature
                
        Returns:
            Dictionary containing:
                - audio: Generated audio as numpy array
                - sample_rate: Audio sample rate
                
        Raises:
            ValidationError: If inputs are invalid.
            ProcessingError: If generation fails.
        """
        try:
            # Validate inputs
            self.validate_inputs(inputs)
            
            # Get parameters
            text = inputs["text"]
            duration = inputs.get("duration", 30.0)  # Default 30 seconds
            temperature = inputs.get("temperature", 1.0)
            
            # Process text
            text_tokens = self.text_processor.tokenize(text)
            text_tokens = self.to_device(text_tokens)
            
            # Generate
            with torch.no_grad():
                # Encode text
                text_embedding = self.text_encoder.embedding(text_tokens)
                encoded = self.text_encoder.transformer(text_embedding)
                
                # Calculate target length
                target_length = int(duration * self.sample_rate)
                
                # Decode to audio
                decoded = self.audio_decoder.upsampler(encoded)
                decoded = decoded.transpose(1, 2)  # [B, C, T]
                audio = self.audio_decoder.conv_stack(decoded)
                
                # Adjust length
                audio = self._adjust_length(audio, target_length)
                
                # Apply temperature
                if temperature != 1.0:
                    audio = audio / temperature
                
                # Convert to numpy
                audio = audio.cpu().numpy().squeeze()
                
            return {
                "audio": audio,
                "sample_rate": self.sample_rate
            }
            
        except Exception as e:
            raise ProcessingError(f"Error in music generation: {e}")
            
    def _adjust_length(
        self,
        audio: torch.Tensor,
        target_length: int
    ) -> torch.Tensor:
        """Adjust audio length to target length.
        
        Args:
            audio: Audio tensor [B, 1, T].
            target_length: Target length in samples.
            
        Returns:
            Adjusted audio tensor.
        """
        current_length = audio.size(-1)
        
        if current_length > target_length:
            # Trim
            audio = audio[..., :target_length]
        elif current_length < target_length:
            # Pad
            padding = target_length - current_length
            audio = torch.nn.functional.pad(audio, (0, padding))
            
        return audio
        
    def preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess model inputs.
        
        Args:
            inputs: Raw input dictionary.
            
        Returns:
            Preprocessed inputs.
        """
        # Normalize text
        if "text" in inputs:
            inputs["text"] = self.text_processor.normalize_text(inputs["text"])
            
        # Validate duration
        if "duration" in inputs:
            duration = float(inputs["duration"])
            if duration <= 0:
                raise ValidationError("Duration must be positive")
            if duration > self.config.get("max_duration", 300):
                raise ValidationError("Duration exceeds maximum allowed")
            inputs["duration"] = duration
            
        # Validate temperature
        if "temperature" in inputs:
            temperature = float(inputs["temperature"])
            if temperature <= 0:
                raise ValidationError("Temperature must be positive")
            inputs["temperature"] = temperature
            
        return inputs
        
    def postprocess_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess model outputs.
        
        Args:
            outputs: Raw output dictionary.
            
        Returns:
            Processed outputs.
        """
        if "audio" in outputs:
            # Normalize audio
            audio = outputs["audio"]
            audio = audio / np.abs(audio).max()
            outputs["audio"] = audio
            
        return outputs
        
    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate model inputs.
        
        Args:
            inputs: Input dictionary.
            
        Raises:
            ValidationError: If inputs are invalid.
        """
        super().validate_inputs(inputs)
        
        # Validate text
        if not inputs.get("text"):
            raise ValidationError("Text input is required")
            
        # Validate text length
        max_length = self.config.get("max_text_length", 1000)
        if len(inputs["text"]) > max_length:
            raise ValidationError(f"Text exceeds maximum length of {max_length}")
            
    def get_memory_usage(self) -> Dict[str, float]:
        """Get model memory usage.
        
        Returns:
            Dictionary of memory usage statistics.
        """
        stats = super().get_memory_usage()
        
        # Add text processor memory
        text_proc_mem = self.text_processor.get_memory_usage()
        stats["text_processor_memory"] = text_proc_mem
        
        return stats 