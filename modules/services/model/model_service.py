"""Service for managing AI models."""
import logging
from typing import Dict, Any, List, Optional

from ...core.base.base_model import BaseAIModel
from ...core.base.model_registry import ModelRegistry
from ...core.config.model_config import ModelConfig
from ...core.exceptions.base_exceptions import ModelNotFoundError
from ...core.resources.base_resources import ModelResource
from ...utils.metrics.prometheus_metrics import MODEL_TRAINING_TIME, TRAINING_LOSS, VALIDATION_LOSS


class ModelService:
    """Service for managing AI models."""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        resource_service: 'ResourceService'
    ):
        """Initialize the model service.
        
        Args:
            model_registry: Registry for trained models.
            resource_service: Service for accessing resources.
        """
        self.model_registry = model_registry
        self.resource_service = resource_service
        self.logger = logging.getLogger(__name__)
        self._model_cache: Dict[str, BaseAIModel] = {}
        
    async def get_model(
        self,
        model_type: str,
        config: Optional[ModelConfig] = None,
        version: Optional[str] = None
    ) -> BaseAIModel:
        """Get a model instance.
        
        Args:
            model_type: Type of model to get.
            config: Optional model configuration.
            version: Optional model version.
            
        Returns:
            Model instance.
            
        Raises:
            ModelNotFoundError: If model not found.
        """
        # Check cache first
        cache_key = f"{model_type}:{version or 'latest'}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
            
        # Get model path from registry
        model_path = self.model_registry.get_model_path(
            model_id=model_type,
            version=version
        )
        
        if not model_path:
            raise ModelNotFoundError(
                f"Model {model_type} not found"
            )
            
        # Get model resource
        model_resource = await self.resource_service.get_resource(
            resource_id=model_type
        )
        
        if not isinstance(model_resource, ModelResource):
            raise ModelNotFoundError(
                f"Resource {model_type} is not a model"
            )
            
        # Create model instance
        model_class = self._get_model_class(model_type)
        model_config = config or ModelConfig(
            model_id=model_type,
            model_type=model_type
        )
        
        model = model_class(model_config)
        model.load(model_path)
        
        # Cache model
        self._model_cache[cache_key] = model
        
        return model
        
    async def list_models(self) -> List[ModelResource]:
        """List all available models.
        
        Returns:
            List of model resources.
        """
        models = []
        for model_id in self.model_registry.index["models"]:
            model = await self.resource_service.get_resource(model_id)
            if isinstance(model, ModelResource):
                models.append(model)
        return models
        
    async def train_model(
        self,
        model_type: str,
        training_config: Dict[str, Any],
        train_data: List[Dict[str, Any]],
        val_data: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Train a new model.
        
        Args:
            model_type: Type of model to train.
            training_config: Training configuration.
            train_data: Training data.
            val_data: Optional validation data.
            
        Returns:
            Model version ID.
        """
        self.logger.info(f"Training model: {model_type}")
        
        # Get model class
        model_class = self._get_model_class(model_type)
        
        # Create model instance
        model_config = ModelConfig(
            model_id=model_type,
            model_type=model_type
        )
        model = model_class(model_config)
        
        # Train model with timing
        with MODEL_TRAINING_TIME.labels(
            model_type=model_type,
            model_id=model_type
        ).time():
            results = await model.train(
                training_config=training_config,
                train_data=train_data,
                val_data=val_data
            )
            
        # Track metrics
        for epoch, loss in enumerate(results["train_losses"]):
            TRAINING_LOSS.labels(
                model_type=model_type,
                model_id=model_type
            ).observe(loss)
            
        if results["val_losses"]:
            for epoch, loss in enumerate(results["val_losses"]):
                VALIDATION_LOSS.labels(
                    model_type=model_type,
                    model_id=model_type
                ).observe(loss)
                
        # Save model
        version = self.model_registry.register_model(
            model_id=model_type,
            model_type=model_type,
            version=str(results["num_epochs"]),
            path=model_config.weights_path,
            metadata={
                "training_config": training_config,
                "results": results
            }
        )
        
        return version
        
    def _get_model_class(self, model_type: str) -> type:
        """Get model class for type.
        
        Args:
            model_type: Type of model.
            
        Returns:
            Model class.
            
        Raises:
            ModelNotFoundError: If model type not found.
        """
        # Import model classes
        from ...models.audio import AudioModel
        from ...models.midi import MIDIModel
        from ...models.text import TextModel
        
        # Map model types to classes
        model_classes = {
            "text_to_music": TextModel,
            "midi_variation": MIDIModel,
            "audio_variation": AudioModel
        }
        
        if model_type not in model_classes:
            raise ModelNotFoundError(
                f"Unknown model type: {model_type}"
            )
            
        return model_classes[model_type] 