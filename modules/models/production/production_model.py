"""Production model for handling music generation in production."""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel

from ...core.exceptions import DiscoMusicaError
from ...core.resources import BaseResource
from ...core.config import ModelConfig

logger = logging.getLogger(__name__)

class ProductionConfig(BaseModel):
    """Configuration for the production model."""
    model_name: str
    model_version: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    max_length: int = 1024
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    do_sample: bool = True
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    metadata: Dict[str, Any] = {}

class ProductionModel:
    """Model for handling music generation in production."""
    
    def __init__(
        self,
        config: ProductionConfig,
        model: Optional[nn.Module] = None
    ):
        """Initialize the production model.
        
        Args:
            config: Model configuration.
            model: Optional pre-trained model.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = self._load_model()
            
    def _load_model(self) -> nn.Module:
        """Load the pre-trained model.
        
        Returns:
            Loaded model.
        """
        try:
            # Load model from registry
            model = ModelRegistry.get_model(
                self.config.model_name,
                self.config.model_version
            )
            return model.to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise DiscoMusicaError(f"Failed to load model: {e}")
            
    async def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate music from input.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Optional attention mask.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated music and metadata.
        """
        try:
            # Move inputs to device
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                
            # Update generation parameters
            gen_kwargs = self.config.dict()
            gen_kwargs.update(kwargs)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )
                
            # Process outputs
            generated_ids = outputs.sequences
            generated_scores = outputs.scores if hasattr(outputs, "scores") else None
            
            # Convert to numpy arrays
            generated_ids = generated_ids.cpu().numpy()
            if generated_scores is not None:
                generated_scores = [score.cpu().numpy() for score in generated_scores]
                
            return {
                "generated_ids": generated_ids,
                "generated_scores": generated_scores,
                "metadata": {
                    "model_name": self.config.model_name,
                    "model_version": self.config.model_version,
                    "generation_params": gen_kwargs,
                    "device": str(self.device)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise DiscoMusicaError(f"Generation failed: {e}")
            
    async def batch_generate(
        self,
        input_ids: List[torch.Tensor],
        attention_masks: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate music from multiple inputs.
        
        Args:
            input_ids: List of input token IDs.
            attention_masks: Optional list of attention masks.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of generated music and metadata.
        """
        try:
            # Pad inputs
            max_length = max(ids.size(0) for ids in input_ids)
            padded_inputs = []
            padded_masks = []
            
            for i, ids in enumerate(input_ids):
                # Pad input IDs
                padding = torch.zeros(
                    max_length - ids.size(0),
                    dtype=ids.dtype,
                    device=self.device
                )
                padded_ids = torch.cat([ids, padding])
                padded_inputs.append(padded_ids)
                
                # Pad attention mask if provided
                if attention_masks is not None:
                    mask = attention_masks[i]
                    padding = torch.zeros(
                        max_length - mask.size(0),
                        dtype=mask.dtype,
                        device=self.device
                    )
                    padded_mask = torch.cat([mask, padding])
                    padded_masks.append(padded_mask)
                    
            # Stack inputs
            input_ids = torch.stack(padded_inputs)
            if attention_masks is not None:
                attention_mask = torch.stack(padded_masks)
            else:
                attention_mask = None
                
            # Generate
            outputs = await self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            
            # Process outputs
            generated_ids = outputs["generated_ids"]
            generated_scores = outputs["generated_scores"]
            
            # Split into individual sequences
            results = []
            for i in range(len(input_ids)):
                result = {
                    "generated_ids": generated_ids[i],
                    "generated_scores": [score[i] for score in generated_scores] if generated_scores is not None else None,
                    "metadata": outputs["metadata"]
                }
                results.append(result)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {e}")
            raise DiscoMusicaError(f"Batch generation failed: {e}")
            
    async def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode input into embeddings.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Optional attention mask.
            
        Returns:
            Input embeddings.
        """
        try:
            # Move inputs to device
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                
            # Encode
            with torch.no_grad():
                outputs = self.model.encode(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
            return outputs.last_hidden_state
            
        except Exception as e:
            self.logger.error(f"Encoding failed: {e}")
            raise DiscoMusicaError(f"Encoding failed: {e}")
            
    async def decode(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Decode embeddings into token IDs.
        
        Args:
            hidden_states: Hidden states to decode.
            attention_mask: Optional attention mask.
            **kwargs: Additional decoding parameters.
            
        Returns:
            Decoded token IDs.
        """
        try:
            # Move inputs to device
            hidden_states = hidden_states.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                
            # Decode
            with torch.no_grad():
                outputs = self.model.decode(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    **kwargs
                )
                
            return outputs.sequences
            
        except Exception as e:
            self.logger.error(f"Decoding failed: {e}")
            raise DiscoMusicaError(f"Decoding failed: {e}")
            
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model.
        """
        try:
            # Save model state
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config.dict()
                },
                path
            )
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise DiscoMusicaError(f"Failed to save model: {e}")
            
    @classmethod
    def load(cls, path: str) -> "ProductionModel":
        """Load a model from disk.
        
        Args:
            path: Path to load the model from.
            
        Returns:
            Loaded model.
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(path)
            
            # Create config
            config = ProductionConfig(**checkpoint["config"])
            
            # Create model
            model = cls(config)
            
            # Load state dict
            model.model.load_state_dict(checkpoint["model_state_dict"])
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise DiscoMusicaError(f"Failed to load model: {e}") 