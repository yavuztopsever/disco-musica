"""Registry for trained models."""
import os
from typing import Dict, List, Optional
import json

from ..resources.base_resources import ModelResource


class ModelRegistry:
    """Registry for trained models."""
    
    def __init__(self, registry_path: str):
        """Initialize the registry.
        
        Args:
            registry_path: Path to model registry directory.
        """
        self.registry_path = registry_path
        self.index_path = os.path.join(registry_path, "index.json")
        os.makedirs(registry_path, exist_ok=True)
        
        # Initialize or load index
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                self.index = json.load(f)
        else:
            self.index = {
                "models": {},
                "latest_versions": {}
            }
            self._save_index()
            
    def register_model(
        self,
        model_id: str,
        model_type: str,
        version: str,
        path: str,
        metadata: Dict = None
    ) -> str:
        """Register a model in the registry.
        
        Args:
            model_id: Unique model identifier.
            model_type: Type of model.
            version: Model version.
            path: Path to model weights.
            metadata: Additional metadata.
            
        Returns:
            Full model ID (model_id:version).
        """
        full_id = f"{model_id}:{version}"
        
        if model_id not in self.index["models"]:
            self.index["models"][model_id] = {}
            
        self.index["models"][model_id][version] = {
            "path": path,
            "model_type": model_type,
            "metadata": metadata or {}
        }
        
        # Update latest version
        self.index["latest_versions"][model_id] = version
        
        self._save_index()
        return full_id
        
    def get_model_path(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Optional[str]:
        """Get path to model weights.
        
        Args:
            model_id: Model identifier.
            version: Optional version (uses latest if None).
            
        Returns:
            Path to model weights, or None if not found.
        """
        if model_id not in self.index["models"]:
            return None
            
        if version is None:
            version = self.index["latest_versions"].get(model_id)
            if version is None:
                return None
                
        if version not in self.index["models"][model_id]:
            return None
            
        return self.index["models"][model_id][version]["path"]
        
    def _save_index(self):
        """Save index to disk."""
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2) 