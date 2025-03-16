"""
Model Service module for Disco Musica.

This module provides services for model selection, management, and registration.
"""

import os
import json
import datetime
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from huggingface_hub import HfApi, model_info, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from modules.core.config import config
from modules.core.resources.resource_manager import ResourceManager
from modules.core.resources.model_resource import ModelResource
from modules.models.model_registry import ModelRegistry
from modules.core.processors.processor_manager import ProcessorManager
from modules.exceptions.base_exceptions import (
    ProcessingError,
    ValidationError,
    ResourceNotFoundError
)


class ModelService:
    """
    Service for model selection, management, and registration.
    
    This class provides functionalities for discovering, downloading, and managing
    models for various music generation tasks.
    """
    
    def __init__(
        self,
        resource_manager: ResourceManager,
        model_registry: ModelRegistry,
        processor_manager: ProcessorManager,
        cache_dir: Union[str, Path]
    ):
        """
        Initialize the ModelService.
        
        Args:
            resource_manager: Resource manager instance.
            model_registry: Model registry instance.
            processor_manager: Processor manager instance.
            cache_dir: Directory for model caching.
        """
        self.resource_manager = resource_manager
        self.model_registry = model_registry
        self.processor_manager = processor_manager
        self.cache_dir = Path(cache_dir)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Model instance cache
        self._model_cache: Dict[str, Any] = {}
        
        self.pretrained_dir = Path(config.get("models", "model_cache_dir", "models")) / "pretrained"
        self.finetuned_dir = Path(config.get("models", "model_cache_dir", "models")) / "finetuned"
        self.model_registry = {}
        self.task_registry = {
            "text_to_music": [],
            "audio_to_music": [],
            "midi_to_audio": [],
            "image_to_music": [],
        }
        
        # Create directories if they don't exist
        os.makedirs(self.pretrained_dir, exist_ok=True)
        os.makedirs(self.finetuned_dir, exist_ok=True)
        
        # Load default models registry
        self._initialize_default_models()
        
        # Scan local models
        self._scan_local_models()
    
    def _initialize_default_models(self) -> None:
        """
        Initialize the registry with default models.
        """
        default_models = {
            "text_to_music": [
                {
                    "id": "facebook/musicgen-small",
                    "name": "MusicGen Small",
                    "description": "Small model for text-to-music generation (300M parameters).",
                    "capabilities": ["text_to_music"],
                    "tags": ["transformer", "text-to-audio", "music"],
                    "source": "huggingface",
                    "size": "small",
                    "model_class": "TextToMusicModel"
                },
                {
                    "id": "facebook/musicgen-medium",
                    "name": "MusicGen Medium",
                    "description": "Medium model for text-to-music generation (1.5B parameters).",
                    "capabilities": ["text_to_music"],
                    "tags": ["transformer", "text-to-audio", "music"],
                    "source": "huggingface",
                    "size": "medium",
                    "model_class": "TextToMusicModel"
                },
                {
                    "id": "facebook/musicgen-large",
                    "name": "MusicGen Large",
                    "description": "Large model for text-to-music generation (3.3B parameters).",
                    "capabilities": ["text_to_music"],
                    "tags": ["transformer", "text-to-audio", "music"],
                    "source": "huggingface",
                    "size": "large",
                    "model_class": "TextToMusicModel"
                }
            ],
            "audio_to_music": [
                {
                    "id": "facebook/musicgen-melody",
                    "name": "MusicGen Melody",
                    "description": "Specialized model for music continuation and accompaniment.",
                    "capabilities": ["text_to_music", "audio_to_music"],
                    "tags": ["transformer", "audio-to-audio", "music", "continuation"],
                    "source": "huggingface",
                    "size": "medium",
                    "model_class": "AudioToMusicModel"
                }
            ],
            "midi_to_audio": [
                {
                    "id": "facebook/musicgen-melody",
                    "name": "MusicGen Melody",
                    "description": "MIDI conditioning model for music generation.",
                    "capabilities": ["text_to_music", "audio_to_music", "midi_to_audio"],
                    "tags": ["transformer", "midi-to-audio", "music"],
                    "source": "huggingface",
                    "size": "medium",
                    "model_class": "MIDIToAudioModel"
                }
            ]
        }
        
        # Register default models
        for task, models in default_models.items():
            for model in models:
                model_id = model["id"]
                self.model_registry[model_id] = model
                self.task_registry[task].append(model_id)
    
    def _scan_local_models(self) -> None:
        """
        Scan local directories for models and update the registry.
        """
        # Scan pretrained models
        for model_dir in self.pretrained_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    model_id = f"local/{model_dir.name}"
                    capabilities = metadata.get("capabilities", [])
                    
                    self.model_registry[model_id] = {
                        "id": model_id,
                        "name": metadata.get("name", model_dir.name),
                        "description": metadata.get("description", "Locally available model"),
                        "capabilities": capabilities,
                        "tags": metadata.get("tags", []),
                        "source": "local",
                        "local_path": str(model_dir),
                        "model_class": metadata.get("model_class", "UnknownModel")
                    }
                    
                    # Add to task registry for each capability
                    for capability in capabilities:
                        if capability in self.task_registry:
                            self.task_registry[capability].append(model_id)
                    
                except Exception as e:
                    print(f"Error loading metadata for {model_dir}: {e}")
        
        # Scan finetuned models
        for model_dir in self.finetuned_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    model_id = f"finetuned/{model_dir.name}"
                    capabilities = metadata.get("capabilities", [])
                    
                    self.model_registry[model_id] = {
                        "id": model_id,
                        "name": metadata.get("name", model_dir.name),
                        "description": metadata.get("description", "Fine-tuned model"),
                        "capabilities": capabilities,
                        "tags": metadata.get("tags", []),
                        "source": "finetuned",
                        "local_path": str(model_dir),
                        "model_class": metadata.get("model_class", "UnknownModel"),
                        "base_model": metadata.get("base_model", None)
                    }
                    
                    # Add to task registry for each capability
                    for capability in capabilities:
                        if capability in self.task_registry:
                            self.task_registry[capability].append(model_id)
                    
                except Exception as e:
                    print(f"Error loading metadata for {model_dir}: {e}")
    
    def get_models_for_task(self, task: str) -> List[Dict]:
        """
        Get available models for a specific task.
        
        Args:
            task: Task name (e.g., 'text_to_music', 'audio_to_music').
            
        Returns:
            List of model dictionaries.
        """
        if task not in self.task_registry:
            return []
        
        models = []
        for model_id in self.task_registry[task]:
            if model_id in self.model_registry:
                models.append(self.model_registry[model_id])
        
        return models
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """
        Get information about a specific model.
        
        Args:
            model_id: ID of the model.
            
        Returns:
            Model information dictionary or None if not found.
        """
        return self.model_registry.get(model_id)
    
    def is_model_available_locally(self, model_id: str) -> bool:
        """
        Check if a model is available locally.
        
        Args:
            model_id: ID of the model.
            
        Returns:
            True if the model is available locally, False otherwise.
        """
        model_info = self.get_model_info(model_id)
        if not model_info:
            return False
        
        if "local_path" in model_info:
            return Path(model_info["local_path"]).exists()
        
        if model_info["source"] == "huggingface":
            model_path = self.pretrained_dir / model_id.split("/")[-1]
            return model_path.exists()
        
        return False
    
    def search_huggingface_models(
        self, 
        query: str, 
        task_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for models on Hugging Face Hub.
        
        Args:
            query: Search query.
            task_type: Task type filter.
            limit: Maximum number of results.
            
        Returns:
            List of model dictionaries.
        """
        try:
            api = HfApi()
            filter_str = ""
            
            if task_type == "text_to_music":
                filter_str = "text-to-audio"
            elif task_type == "audio_to_music":
                filter_str = "audio-to-audio"
            
            models = api.list_models(
                search=query,
                filter=filter_str if filter_str else None,
                limit=limit
            )
            
            results = []
            for model in models:
                model_id = model.id
                
                # Skip if already in registry
                if model_id in self.model_registry:
                    continue
                
                # Get model info
                try:
                    info = model_info(model_id)
                    tags = model.tags or []
                    pipeline_tag = model.pipeline_tag or ""
                    
                    # Determine capabilities based on tags and pipeline
                    capabilities = []
                    if "text-to-audio" in tags or pipeline_tag == "text-to-audio":
                        capabilities.append("text_to_music")
                    if "audio-to-audio" in tags or pipeline_tag == "audio-to-audio":
                        capabilities.append("audio_to_music")
                    if "midi" in tags:
                        capabilities.append("midi_to_audio")
                    
                    # Skip if no relevant capabilities
                    if not capabilities:
                        continue
                    
                    # Determine model class based on capabilities
                    model_class = "TextToMusicModel"
                    if "audio_to_music" in capabilities and "text_to_music" not in capabilities:
                        model_class = "AudioToMusicModel"
                    if "midi_to_audio" in capabilities and "text_to_music" not in capabilities:
                        model_class = "MIDIToAudioModel"
                    
                    # Add to results
                    results.append({
                        "id": model_id,
                        "name": model_id.split("/")[-1],
                        "description": info.description or "No description available",
                        "capabilities": capabilities,
                        "tags": tags,
                        "source": "huggingface",
                        "model_class": model_class,
                        "downloads": info.downloads,
                        "likes": info.likes
                    })
                except Exception as e:
                    print(f"Error getting info for {model_id}: {e}")
            
            return results
        
        except Exception as e:
            print(f"Error searching Hugging Face models: {e}")
            return []
    
    def download_model(self, model_id: str) -> Optional[str]:
        """
        Download a model from Hugging Face Hub.
        
        Args:
            model_id: ID of the model.
            
        Returns:
            Path to the downloaded model or None if download failed.
        """
        if self.is_model_available_locally(model_id):
            model_info = self.get_model_info(model_id)
            if "local_path" in model_info:
                return model_info["local_path"]
            else:
                return str(self.pretrained_dir / model_id.split("/")[-1])
        
        try:
            # Get model info
            model_info = self.get_model_info(model_id)
            if not model_info:
                # Try to get info from Hugging Face
                try:
                    info = model_info(model_id)
                    tags = info.tags or []
                    pipeline_tag = info.pipeline_tag or ""
                    
                    # Determine capabilities based on tags and pipeline
                    capabilities = []
                    if "text-to-audio" in tags or pipeline_tag == "text-to-audio":
                        capabilities.append("text_to_music")
                    if "audio-to-audio" in tags or pipeline_tag == "audio-to-audio":
                        capabilities.append("audio_to_music")
                    if "midi" in tags:
                        capabilities.append("midi_to_audio")
                    
                    # Determine model class based on capabilities
                    model_class = "TextToMusicModel"
                    if "audio_to_music" in capabilities and "text_to_music" not in capabilities:
                        model_class = "AudioToMusicModel"
                    if "midi_to_audio" in capabilities and "text_to_music" not in capabilities:
                        model_class = "MIDIToAudioModel"
                    
                    model_info = {
                        "id": model_id,
                        "name": model_id.split("/")[-1],
                        "description": info.description or "No description available",
                        "capabilities": capabilities,
                        "tags": tags,
                        "source": "huggingface",
                        "model_class": model_class
                    }
                    
                    # Add to registry
                    self.model_registry[model_id] = model_info
                    
                    # Add to task registry for each capability
                    for capability in capabilities:
                        if capability in self.task_registry:
                            self.task_registry[capability].append(model_id)
                    
                except Exception as e:
                    print(f"Error getting info for {model_id}: {e}")
                    return None
            
            # Get local path
            local_path = self.pretrained_dir / model_id.split("/")[-1]
            os.makedirs(local_path, exist_ok=True)
            
            # Download model
            print(f"Downloading model {model_id}...")
            snapshot_download(
                repo_id=model_id,
                local_dir=local_path,
                local_dir_use_symlinks=False
            )
            
            # Update model info with local path
            model_info["local_path"] = str(local_path)
            self.model_registry[model_id] = model_info
            
            # Create metadata file
            metadata = {
                "name": model_info.get("name", model_id.split("/")[-1]),
                "description": model_info.get("description", "Downloaded from Hugging Face Hub"),
                "capabilities": model_info.get("capabilities", []),
                "tags": model_info.get("tags", []),
                "source": "huggingface",
                "source_repo": model_id,
                "model_class": model_info.get("model_class", "UnknownModel"),
                "download_date": datetime.datetime.now().isoformat()
            }
            
            with open(local_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Model {model_id} downloaded to {local_path}")
            return str(local_path)
        
        except Exception as e:
            print(f"Error downloading model {model_id}: {e}")
            return None
    
    def get_default_model_for_task(self, task: str) -> Optional[str]:
        """
        Get the default model ID for a specific task.
        
        Args:
            task: Task name (e.g., 'text_to_music', 'audio_to_music').
            
        Returns:
            Model ID or None if no default model is found.
        """
        if task == "text_to_music":
            return config.get("models", "default_text_to_music", "facebook/musicgen-small")
        elif task == "audio_to_music":
            return config.get("models", "default_audio_to_audio", "facebook/musicgen-melody")
        elif task == "midi_to_audio":
            return config.get("models", "default_midi_to_audio", "facebook/musicgen-melody")
        elif task == "image_to_music":
            return config.get("models", "default_image_to_music", None)
        
        return None
    
    def register_model(self, model_info: Dict) -> bool:
        """
        Register a new model in the registry.
        
        Args:
            model_info: Model information dictionary.
            
        Returns:
            True if registration succeeded, False otherwise.
        """
        model_id = model_info.get("id")
        if not model_id:
            print("Model ID is required")
            return False
        
        capabilities = model_info.get("capabilities", [])
        if not capabilities:
            print("At least one capability is required")
            return False
        
        # Add to model registry
        self.model_registry[model_id] = model_info
        
        # Add to task registry for each capability
        for capability in capabilities:
            if capability in self.task_registry:
                if model_id not in self.task_registry[capability]:
                    self.task_registry[capability].append(model_id)
        
        return True
    
    def create_model_instance(self, model_id: str):
        """
        Create a model instance for the specified model ID.
        
        Args:
            model_id: ID of the model.
            
        Returns:
            Model instance or None if creation failed.
        """
        model_info = self.get_model_info(model_id)
        if not model_info:
            print(f"Model {model_id} not found in registry")
            return None
        
        model_class_name = model_info.get("model_class", "TextToMusicModel")
        model_type = "finetuned" if model_info.get("source") == "finetuned" else "pretrained"
        
        try:
            # Import the model class
            if model_class_name == "TextToMusicModel":
                from modules.core.text_to_music_model import TextToMusicModel
                return TextToMusicModel(model_id, model_type)
            
            elif model_class_name == "AudioToMusicModel":
                from modules.core.audio_to_music_model import AudioToMusicModel
                return AudioToMusicModel(model_id, model_type)
            
            elif model_class_name == "MIDIToAudioModel":
                from modules.core.midi_to_audio_model import MIDIToAudioModel
                return MIDIToAudioModel(model_id, model_type)
            
            else:
                print(f"Unknown model class: {model_class_name}")
                return None
        
        except Exception as e:
            print(f"Error creating model instance for {model_id}: {e}")
            return None

    async def load_model(
        self,
        model_id: str,
        force_reload: bool = False
    ) -> Any:
        """Load a model instance.
        
        Args:
            model_id: Model ID.
            force_reload: Whether to force reload from disk.
            
        Returns:
            Model instance.
            
        Raises:
            ResourceNotFoundError: If model not found.
            ProcessingError: If loading fails.
        """
        try:
            # Check cache first
            if not force_reload and model_id in self._model_cache:
                return self._model_cache[model_id]
                
            # Get model resource
            model = self.resource_manager.get_resource("model", model_id)
            
            # Get model instance
            model_instance = self.model_registry.get_model(
                model.model_type,
                model.version
            )
            
            # Load weights if available
            if model.weights_path:
                await model_instance.load(model.weights_path)
                
            # Cache instance
            self._model_cache[model_id] = model_instance
            return model_instance
            
        except Exception as e:
            raise ProcessingError(f"Error loading model: {e}")
            
    async def unload_model(self, model_id: str) -> None:
        """Unload a model instance.
        
        Args:
            model_id: Model ID.
            
        Raises:
            ResourceNotFoundError: If model not found.
            ProcessingError: If unloading fails.
        """
        try:
            if model_id in self._model_cache:
                model_instance = self._model_cache[model_id]
                await model_instance.unload()
                del self._model_cache[model_id]
                
        except Exception as e:
            raise ProcessingError(f"Error unloading model: {e}")
            
    async def train_model(
        self,
        model_id: str,
        train_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> ModelResource:
        """Train a model.
        
        Args:
            model_id: Model ID.
            train_data: Training data.
            val_data: Optional validation data.
            config: Optional training configuration.
            
        Returns:
            Updated model resource.
            
        Raises:
            ResourceNotFoundError: If model not found.
            ProcessingError: If training fails.
        """
        try:
            # Get model resource
            model = self.resource_manager.get_resource("model", model_id)
            
            # Get model instance
            model_instance = await self.load_model(model_id, force_reload=True)
            
            # Train model
            try:
                metrics = await model_instance.train(
                    train_data,
                    val_data=val_data,
                    **config or {}
                )
                
                # Save weights
                weights_path = self.cache_dir / f"{model_id}.pt"
                await model_instance.save(weights_path)
                
                # Update model resource
                model.set_weights_path(str(weights_path))
                model.set_training_config(config or {})
                model.update_metrics(metrics)
                model.set_last_trained(datetime.datetime.utcnow())
                
                return model
                
            except Exception as e:
                raise ProcessingError(f"Training failed: {e}")
                
        except Exception as e:
            raise ProcessingError(f"Error in model training: {e}")
            
    async def evaluate_model(
        self,
        model_id: str,
        test_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Evaluate a model.
        
        Args:
            model_id: Model ID.
            test_data: Test data.
            config: Optional evaluation configuration.
            
        Returns:
            Dictionary of evaluation metrics.
            
        Raises:
            ResourceNotFoundError: If model not found.
            ProcessingError: If evaluation fails.
        """
        try:
            # Get model instance
            model_instance = await self.load_model(model_id)
            
            # Evaluate model
            metrics = await model_instance.evaluate(
                test_data,
                **config or {}
            )
            
            # Update model metrics
            model = self.resource_manager.get_resource("model", model_id)
            model.update_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            raise ProcessingError(f"Error in model evaluation: {e}")
            
    def get_default_model(
        self,
        model_type: str
    ) -> Optional[ModelResource]:
        """Get default model for a type.
        
        Args:
            model_type: Model type.
            
        Returns:
            Default model resource or None if not found.
        """
        # Get active models of type
        models = [
            m for m in self.resource_manager.list_resources("model", status="active")
            if m.model_type == model_type
        ]
        
        if not models:
            return None
            
        # Return model with highest metrics
        return max(
            models,
            key=lambda m: sum(m.metrics.values()) if m.metrics else 0
        )
        
    async def cleanup(self) -> None:
        """Clean up model service.
        
        This unloads all models and clears the cache.
        """
        # Unload all models
        for model_id in list(self._model_cache.keys()):
            await self.unload_model(model_id)
            
        # Clear cache
        self._model_cache.clear()

    async def train_generation_model(
        self,
        training_config: Dict[str, Any],
        model_id: Optional[str] = None
    ) -> ModelResource:
        """Train or fine-tune generation model.
        
        Args:
            training_config: Training configuration:
                - data_path: Path to training data
                - model_type: Type of model to train
                - hyperparameters: Training hyperparameters
                - compute_config: Compute configuration
            model_id: Optional model ID for fine-tuning.
            
        Returns:
            Trained model resource.
            
        Raises:
            ResourceNotFoundError: If model_id provided but not found.
            ProcessingError: If training fails.
        """
        try:
            # Prepare training data
            train_data = await self._prepare_training_data(
                training_config["data_path"],
                training_config["model_type"]
            )
            
            # Get or create model
            if model_id:
                model = await self.model_repository.find_by_id(model_id)
                if not model:
                    raise ResourceNotFoundError(f"Model {model_id} not found")
            else:
                model = await self._create_generation_model(training_config)
                
            # Train model
            trained_model = await self.training_service.train_model(
                model=model,
                train_data=train_data,
                hyperparameters=training_config["hyperparameters"],
                compute_config=training_config["compute_config"]
            )
            
            # Save trained model
            await self.model_repository.save(trained_model)
            
            return trained_model
            
        except Exception as e:
            raise ProcessingError(f"Failed to train generation model: {str(e)}")
            
    async def train_production_model(
        self,
        training_config: Dict[str, Any],
        model_id: Optional[str] = None
    ) -> ModelResource:
        """Train or fine-tune production model.
        
        Args:
            training_config: Training configuration:
                - data_path: Path to training data
                - effect_types: Types of effects to model
                - hyperparameters: Training hyperparameters
                - compute_config: Compute configuration
            model_id: Optional model ID for fine-tuning.
            
        Returns:
            Trained model resource.
            
        Raises:
            ResourceNotFoundError: If model_id provided but not found.
            ProcessingError: If training fails.
        """
        try:
            # Prepare training data
            train_data = await self._prepare_effect_training_data(
                training_config["data_path"],
                training_config["effect_types"]
            )
            
            # Get or create model
            if model_id:
                model = await self.model_repository.find_by_id(model_id)
                if not model:
                    raise ResourceNotFoundError(f"Model {model_id} not found")
            else:
                model = await self._create_production_model(training_config)
                
            # Train model
            trained_model = await self.training_service.train_model(
                model=model,
                train_data=train_data,
                hyperparameters=training_config["hyperparameters"],
                compute_config=training_config["compute_config"]
            )
            
            # Save trained model
            await self.model_repository.save(trained_model)
            
            return trained_model
            
        except Exception as e:
            raise ProcessingError(f"Failed to train production model: {str(e)}")
            
    async def train_mastering_model(
        self,
        training_config: Dict[str, Any],
        model_id: Optional[str] = None
    ) -> ModelResource:
        """Train or fine-tune mastering model.
        
        Args:
            training_config: Training configuration:
                - data_path: Path to training data
                - target_metrics: Target audio quality metrics
                - hyperparameters: Training hyperparameters
                - compute_config: Compute configuration
            model_id: Optional model ID for fine-tuning.
            
        Returns:
            Trained model resource.
            
        Raises:
            ResourceNotFoundError: If model_id provided but not found.
            ProcessingError: If training fails.
        """
        try:
            # Prepare training data
            train_data = await self._prepare_mastering_training_data(
                training_config["data_path"],
                training_config["target_metrics"]
            )
            
            # Get or create model
            if model_id:
                model = await self.model_repository.find_by_id(model_id)
                if not model:
                    raise ResourceNotFoundError(f"Model {model_id} not found")
            else:
                model = await self._create_mastering_model(training_config)
                
            # Train model
            trained_model = await self.training_service.train_model(
                model=model,
                train_data=train_data,
                hyperparameters=training_config["hyperparameters"],
                compute_config=training_config["compute_config"]
            )
            
            # Save trained model
            await self.model_repository.save(trained_model)
            
            return trained_model
            
        except Exception as e:
            raise ProcessingError(f"Failed to train mastering model: {str(e)}")
            
    async def _prepare_training_data(
        self,
        data_path: str,
        model_type: str
    ) -> Dict[str, Any]:
        """Prepare training data for generation model.
        
        Args:
            data_path: Path to training data.
            model_type: Type of model to train.
            
        Returns:
            Processed training data.
            
        Raises:
            ProcessingError: If preparation fails.
        """
        try:
            # Load and validate data
            raw_data = await self._load_training_data(data_path)
            
            # Process based on model type
            if model_type == "midi":
                processed_data = await self._process_midi_training_data(raw_data)
            elif model_type == "audio":
                processed_data = await self._process_audio_training_data(raw_data)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            return processed_data
            
        except Exception as e:
            raise ProcessingError(f"Failed to prepare training data: {str(e)}")
            
    async def _prepare_effect_training_data(
        self,
        data_path: str,
        effect_types: List[str]
    ) -> Dict[str, Any]:
        """Prepare training data for production model.
        
        Args:
            data_path: Path to training data.
            effect_types: Types of effects to model.
            
        Returns:
            Processed training data.
            
        Raises:
            ProcessingError: If preparation fails.
        """
        try:
            # Load and validate data
            raw_data = await self._load_training_data(data_path)
            
            # Extract effect parameters and audio features
            processed_data = {
                "input_features": [],
                "effect_parameters": [],
                "output_features": []
            }
            
            for sample in raw_data:
                # Process input audio
                input_features = await self._extract_audio_features(
                    sample["input_audio"]
                )
                
                # Extract effect parameters
                effect_params = await self._extract_effect_parameters(
                    sample["effect_chain"],
                    effect_types
                )
                
                # Process output audio
                output_features = await self._extract_audio_features(
                    sample["output_audio"]
                )
                
                processed_data["input_features"].append(input_features)
                processed_data["effect_parameters"].append(effect_params)
                processed_data["output_features"].append(output_features)
                
            return processed_data
            
        except Exception as e:
            raise ProcessingError(f"Failed to prepare effect training data: {str(e)}")
            
    async def _prepare_mastering_training_data(
        self,
        data_path: str,
        target_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare training data for mastering model.
        
        Args:
            data_path: Path to training data.
            target_metrics: Target audio quality metrics.
            
        Returns:
            Processed training data.
            
        Raises:
            ProcessingError: If preparation fails.
        """
        try:
            # Load and validate data
            raw_data = await self._load_training_data(data_path)
            
            # Process mastering pairs
            processed_data = {
                "input_features": [],
                "target_features": [],
                "quality_metrics": []
            }
            
            for sample in raw_data:
                # Process input mix
                input_features = await self._extract_mastering_features(
                    sample["input_mix"]
                )
                
                # Process reference master
                target_features = await self._extract_mastering_features(
                    sample["reference_master"]
                )
                
                # Calculate quality metrics
                quality_metrics = await self._calculate_quality_metrics(
                    sample["reference_master"],
                    target_metrics
                )
                
                processed_data["input_features"].append(input_features)
                processed_data["target_features"].append(target_features)
                processed_data["quality_metrics"].append(quality_metrics)
                
            return processed_data
            
        except Exception as e:
            raise ProcessingError(f"Failed to prepare mastering training data: {str(e)}")
            
    async def _create_generation_model(
        self,
        config: Dict[str, Any]
    ) -> ModelResource:
        """Create new generation model.
        
        Args:
            config: Model configuration.
            
        Returns:
            Created model resource.
            
        Raises:
            ProcessingError: If creation fails.
        """
        try:
            model = ModelResource(
                resource_id=f"model_{uuid.uuid4().hex}",
                resource_type="generation_model",
                model_info={
                    "type": config["model_type"],
                    "architecture": config.get("architecture", "transformer"),
                    "parameters": config.get("model_parameters", {})
                }
            )
            
            await self.model_repository.save(model)
            return model
            
        except Exception as e:
            raise ProcessingError(f"Failed to create generation model: {str(e)}")
            
    async def _create_production_model(
        self,
        config: Dict[str, Any]
    ) -> ModelResource:
        """Create new production model.
        
        Args:
            config: Model configuration.
            
        Returns:
            Created model resource.
            
        Raises:
            ProcessingError: If creation fails.
        """
        try:
            model = ModelResource(
                resource_id=f"model_{uuid.uuid4().hex}",
                resource_type="production_model",
                model_info={
                    "effect_types": config["effect_types"],
                    "architecture": config.get("architecture", "neural_network"),
                    "parameters": config.get("model_parameters", {})
                }
            )
            
            await self.model_repository.save(model)
            return model
            
        except Exception as e:
            raise ProcessingError(f"Failed to create production model: {str(e)}")
            
    async def _create_mastering_model(
        self,
        config: Dict[str, Any]
    ) -> ModelResource:
        """Create new mastering model.
        
        Args:
            config: Model configuration.
            
        Returns:
            Created model resource.
            
        Raises:
            ProcessingError: If creation fails.
        """
        try:
            model = ModelResource(
                resource_id=f"model_{uuid.uuid4().hex}",
                resource_type="mastering_model",
                model_info={
                    "target_metrics": config["target_metrics"],
                    "architecture": config.get("architecture", "neural_network"),
                    "parameters": config.get("model_parameters", {})
                }
            )
            
            await self.model_repository.save(model)
            return model
            
        except Exception as e:
            raise ProcessingError(f"Failed to create mastering model: {str(e)}")


# Create a global model service instance
model_service = ModelService()


def get_model_service() -> ModelService:
    """
    Get the global model service instance.
    
    Returns:
        ModelService instance.
    """
    global model_service
    return model_service