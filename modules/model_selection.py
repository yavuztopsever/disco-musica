"""
Model Selection Module

This module allows users to select pre-trained models for fine-tuning or inference
and provides access to a curated selection of models from platforms like Hugging Face Hub.
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from huggingface_hub import HfApi, snapshot_download


class ModelSelectionModule:
    """
    A class for model selection operations.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize the ModelSelectionModule.

        Args:
            models_dir: Directory to store the models.
        """
        self.models_dir = Path(models_dir)
        self.pretrained_dir = self.models_dir / "pretrained"
        self.finetuned_dir = self.models_dir / "finetuned"
        
        # Create directories if they don't exist
        os.makedirs(self.pretrained_dir, exist_ok=True)
        os.makedirs(self.finetuned_dir, exist_ok=True)
        
        # Initialize Hugging Face API
        self.hf_api = HfApi()

    def list_available_models(
        self, model_type: Optional[str] = None, task: Optional[str] = None, 
        limit: int = 20
    ) -> List[Dict]:
        """
        List available models from Hugging Face Hub.

        Args:
            model_type: Type of model to list ('audio', 'music', etc.).
            task: Task to filter by ('text-to-music', 'audio-to-audio', etc.).
            limit: Maximum number of models to list.

        Returns:
            List of model dictionaries.
        """
        # This is a placeholder for Hugging Face Hub integration
        # In a real implementation, this would query the Hugging Face Hub API
        
        # Determine filter
        filter = ""
        if model_type:
            filter += f"model-type:{model_type} "
        if task:
            filter += f"task:{task} "
        
        try:
            # Get models from Hugging Face Hub
            models = self.hf_api.list_models(filter=filter, limit=limit)
            
            # Convert models to list of dictionaries
            return [
                {
                    "id": model.id,
                    "name": model.id.split("/")[-1],
                    "author": model.id.split("/")[0],
                    "tags": model.tags,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "pipeline_tag": model.pipeline_tag
                }
                for model in models
            ]
        except Exception as e:
            print(f"Failed to list models from Hugging Face Hub: {e}")
            return []

    def search_models(
        self, query: str, limit: int = 20
    ) -> List[Dict]:
        """
        Search for models on Hugging Face Hub.

        Args:
            query: Search query.
            limit: Maximum number of models to return.

        Returns:
            List of model dictionaries.
        """
        try:
            # Search models on Hugging Face Hub
            models = self.hf_api.list_models(search=query, limit=limit)
            
            # Convert models to list of dictionaries
            return [
                {
                    "id": model.id,
                    "name": model.id.split("/")[-1],
                    "author": model.id.split("/")[0],
                    "tags": model.tags,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "pipeline_tag": model.pipeline_tag
                }
                for model in models
            ]
        except Exception as e:
            print(f"Failed to search models on Hugging Face Hub: {e}")
            return []

    def download_model(
        self, model_id: str, model_type: str = "pretrained"
    ) -> Path:
        """
        Download a model from Hugging Face Hub.

        Args:
            model_id: ID of the model on Hugging Face Hub.
            model_type: Type of model ('pretrained' or 'finetuned').

        Returns:
            Path to the downloaded model.
        """
        # Determine model directory
        if model_type == "pretrained":
            model_dir = self.pretrained_dir / model_id.split("/")[-1]
        elif model_type == "finetuned":
            model_dir = self.finetuned_dir / model_id.split("/")[-1]
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        # Download model
        try:
            # Download model from Hugging Face Hub
            snapshot_download(
                repo_id=model_id,
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"Downloaded model {model_id} to {model_dir}")
            return model_dir
        except Exception as e:
            raise RuntimeError(f"Failed to download model {model_id}: {e}")

    def list_local_models(
        self, model_type: Optional[str] = None
    ) -> List[Dict]:
        """
        List locally available models.

        Args:
            model_type: Type of model to list ('pretrained' or 'finetuned').

        Returns:
            List of model dictionaries.
        """
        models = []
        
        # Determine model directories to search
        if model_type == "pretrained":
            model_dirs = [self.pretrained_dir]
        elif model_type == "finetuned":
            model_dirs = [self.finetuned_dir]
        else:
            model_dirs = [self.pretrained_dir, self.finetuned_dir]
        
        # Search for models
        for model_dir in model_dirs:
            for path in model_dir.iterdir():
                if path.is_dir():
                    # Check if this is a valid model directory
                    config_path = path / "config.json"
                    if config_path.exists():
                        # Get model information
                        with open(config_path, "r") as f:
                            config = json.load(f)
                        
                        # Add model to list
                        models.append({
                            "name": path.name,
                            "path": str(path),
                            "type": "pretrained" if path.parent == self.pretrained_dir else "finetuned",
                            "config": config
                        })
        
        return models

    def get_model_info(
        self, model_name: str, model_type: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get information about a local model.

        Args:
            model_name: Name of the model.
            model_type: Type of model ('pretrained' or 'finetuned').

        Returns:
            Model information dictionary or None if the model is not found.
        """
        # Determine model directories to search
        if model_type == "pretrained":
            model_dirs = [self.pretrained_dir]
        elif model_type == "finetuned":
            model_dirs = [self.finetuned_dir]
        else:
            model_dirs = [self.pretrained_dir, self.finetuned_dir]
        
        # Search for the model
        for model_dir in model_dirs:
            model_path = model_dir / model_name
            if model_path.exists():
                # Check if this is a valid model directory
                config_path = model_path / "config.json"
                if config_path.exists():
                    # Get model information
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    
                    # Return model information
                    return {
                        "name": model_name,
                        "path": str(model_path),
                        "type": "pretrained" if model_dir == self.pretrained_dir else "finetuned",
                        "config": config
                    }
        
        return None

    def delete_model(
        self, model_name: str, model_type: Optional[str] = None
    ) -> bool:
        """
        Delete a local model.

        Args:
            model_name: Name of the model.
            model_type: Type of model ('pretrained' or 'finetuned').

        Returns:
            True if the model was deleted, False otherwise.
        """
        import shutil
        
        # Determine model directories to search
        if model_type == "pretrained":
            model_dirs = [self.pretrained_dir]
        elif model_type == "finetuned":
            model_dirs = [self.finetuned_dir]
        else:
            model_dirs = [self.pretrained_dir, self.finetuned_dir]
        
        # Search for the model
        for model_dir in model_dirs:
            model_path = model_dir / model_name
            if model_path.exists():
                # Delete the model directory
                shutil.rmtree(model_path)
                print(f"Deleted model: {model_path}")
                return True
        
        print(f"Model not found: {model_name}")
        return False

    def get_recommended_models(
        self, task: str
    ) -> List[Dict]:
        """
        Get recommended models for a specific task.

        Args:
            task: Task to get recommendations for.

        Returns:
            List of recommended model dictionaries.
        """
        # This is a placeholder for model recommendations
        # In a real implementation, this would provide recommendations based on the task
        
        # Define recommended models for each task
        recommended_models = {
            "text-to-music": [
                {"id": "facebook/musicgen-small", "name": "MusicGen Small", "description": "Small model for text-to-music generation."},
                {"id": "facebook/musicgen-medium", "name": "MusicGen Medium", "description": "Medium model for text-to-music generation."},
                {"id": "facebook/musicgen-large", "name": "MusicGen Large", "description": "Large model for text-to-music generation."}
            ],
            "audio-to-audio": [
                {"id": "facebook/musicgen-small", "name": "MusicGen Small", "description": "Small model for audio-to-audio generation."},
                {"id": "facebook/musicgen-medium", "name": "MusicGen Medium", "description": "Medium model for audio-to-audio generation."}
            ],
            "midi-to-audio": [
                {"id": "github/midi-ddsp", "name": "MIDI-DDSP", "description": "Model for MIDI-to-audio generation."}
            ],
            "image-to-music": [
                {"id": "misc/imaginemusic", "name": "ImagineMusic", "description": "Model for image-to-music generation."}
            ]
        }
        
        # Return recommended models for the specified task
        if task in recommended_models:
            return recommended_models[task]
        else:
            return []

    def check_model_compatibility(
        self, model_name: str, task: str
    ) -> Dict:
        """
        Check if a model is compatible with a specific task.

        Args:
            model_name: Name of the model.
            task: Task to check compatibility for.

        Returns:
            Compatibility information dictionary.
        """
        # This is a placeholder for model compatibility checking
        # In a real implementation, this would check if the model is compatible with the task
        
        # Get model information
        model_info = self.get_model_info(model_name)
        
        if model_info is None:
            return {"compatible": False, "reason": "Model not found"}
        
        # Check task compatibility
        config = model_info["config"]
        
        # This is a simplified compatibility check
        if task == "text-to-music" and "text_encoder" in config:
            return {"compatible": True}
        elif task == "audio-to-audio" and "audio_encoder" in config:
            return {"compatible": True}
        elif task == "midi-to-audio" and "midi_encoder" in config:
            return {"compatible": True}
        elif task == "image-to-music" and "image_encoder" in config:
            return {"compatible": True}
        
        return {"compatible": False, "reason": f"Model is not compatible with the {task} task"}


# Example usage
if __name__ == "__main__":
    model_selection = ModelSelectionModule()
    # Example: model_selection.list_available_models(model_type="audio", task="text-to-music")