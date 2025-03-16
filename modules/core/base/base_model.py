"""Base model class for AI models."""
import logging
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn

from ..config.model_config import ModelConfig, TrainingConfig
from ..exceptions.base_exceptions import ProcessingError


class BaseAIModel(nn.Module):
    """Base class for AI models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the base model.
        
        Args:
            config: Model configuration.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device = torch.device(config.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
            
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError
        
    async def predict(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run inference on the model.
        
        Args:
            inputs: Dictionary of input tensors or values.
            
        Returns:
            Dictionary of output tensors or values.
            
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError
        
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
            
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError
        
    def save(self, path: str) -> None:
        """Save model weights to path.
        
        Args:
            path: Path to save model weights.
            
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError
        
    def load(self, path: str) -> None:
        """Load model weights from path.
        
        Args:
            path: Path to model weights file.
            
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError
        
    def _prepare_input(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare input tensors.
        
        Args:
            inputs: Dictionary of input tensors or values.
            
        Returns:
            Dictionary of prepared input tensors.
        """
        prepared = {}
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            elif isinstance(value, (list, tuple)):
                prepared[key] = torch.tensor(
                    value,
                    dtype=torch.float32,
                    device=self.device
                )
            elif isinstance(value, (int, float)):
                prepared[key] = torch.tensor(
                    [value],
                    dtype=torch.float32,
                    device=self.device
                )
            else:
                prepared[key] = value
                
        return prepared
        
    def _prepare_output(
        self,
        output: torch.Tensor
    ) -> Dict[str, Any]:
        """Prepare output tensor.
        
        Args:
            output: Output tensor.
            
        Returns:
            Dictionary of prepared output values.
        """
        # Move to CPU and convert to numpy
        output = output.cpu().detach().numpy()
        
        # Remove batch dimension if present
        if output.ndim > 1 and output.shape[0] == 1:
            output = output[0]
            
        return {
            "output": output
        }
        
    def _validate_inputs(
        self,
        inputs: Dict[str, Any],
        required_keys: List[str]
    ) -> None:
        """Validate input dictionary.
        
        Args:
            inputs: Dictionary of input tensors or values.
            required_keys: List of required keys.
            
        Raises:
            ProcessingError: If required keys are missing.
        """
        missing_keys = [
            key for key in required_keys
            if key not in inputs
        ]
        
        if missing_keys:
            raise ProcessingError(
                f"Missing required inputs: {missing_keys}"
            )
            
    def _validate_outputs(
        self,
        outputs: Dict[str, Any],
        required_keys: List[str]
    ) -> None:
        """Validate output dictionary.
        
        Args:
            outputs: Dictionary of output tensors or values.
            required_keys: List of required keys.
            
        Raises:
            ProcessingError: If required keys are missing.
        """
        missing_keys = [
            key for key in required_keys
            if key not in outputs
        ]
        
        if missing_keys:
            raise ProcessingError(
                f"Missing required outputs: {missing_keys}"
            ) 