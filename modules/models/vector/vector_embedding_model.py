"""Vector embedding model for handling vector embeddings."""

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

class VectorEmbeddingConfig(BaseModel):
    """Configuration for the vector embedding model."""
    model_name: str
    model_version: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_dim: int = 768
    max_length: int = 512
    batch_size: int = 32
    normalize_embeddings: bool = True
    metadata: Dict[str, Any] = {}

class VectorEmbeddingModel:
    """Model for handling vector embeddings."""
    
    def __init__(
        self,
        config: VectorEmbeddingConfig,
        model: Optional[nn.Module] = None
    ):
        """Initialize the vector embedding model.
        
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
            
    async def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Encode input into embeddings.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Optional attention mask.
            **kwargs: Additional encoding parameters.
            
        Returns:
            Encoded embeddings and metadata.
        """
        try:
            # Move inputs to device
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                
            # Update encoding parameters
            encode_kwargs = self.config.dict()
            encode_kwargs.update(kwargs)
            
            # Encode
            with torch.no_grad():
                outputs = self.model.encode(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **encode_kwargs
                )
                
            # Process outputs
            embeddings = outputs.last_hidden_state
            if self.config.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                
            # Convert to numpy arrays
            embeddings = embeddings.cpu().numpy()
            
            return {
                "embeddings": embeddings,
                "metadata": {
                    "model_name": self.config.model_name,
                    "model_version": self.config.model_version,
                    "encoding_params": encode_kwargs,
                    "device": str(self.device)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Encoding failed: {e}")
            raise DiscoMusicaError(f"Encoding failed: {e}")
            
    async def batch_encode(
        self,
        input_ids_list: List[torch.Tensor],
        attention_masks: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Encode multiple inputs into embeddings.
        
        Args:
            input_ids_list: List of input token IDs.
            attention_masks: Optional list of attention masks.
            **kwargs: Additional encoding parameters.
            
        Returns:
            List of encoded embeddings and metadata.
        """
        try:
            results = []
            for i, input_ids in enumerate(input_ids_list):
                # Get attention mask if provided
                attention_mask = attention_masks[i] if attention_masks is not None else None
                
                # Encode input
                result = await self.encode(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
                results.append(result)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Batch encoding failed: {e}")
            raise DiscoMusicaError(f"Batch encoding failed: {e}")
            
    async def compute_similarity(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        similarity_type: str = "cosine"
    ) -> torch.Tensor:
        """Compute similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings.
            embeddings2: Second set of embeddings.
            similarity_type: Type of similarity to compute ("cosine", "dot", "l2").
            
        Returns:
            Similarity matrix.
        """
        try:
            # Move embeddings to device
            embeddings1 = embeddings1.to(self.device)
            embeddings2 = embeddings2.to(self.device)
            
            # Compute similarity
            if similarity_type == "cosine":
                similarity = torch.nn.functional.cosine_similarity(
                    embeddings1.unsqueeze(1),
                    embeddings2.unsqueeze(0),
                    dim=-1
                )
            elif similarity_type == "dot":
                similarity = torch.matmul(embeddings1, embeddings2.t())
            elif similarity_type == "l2":
                # Compute L2 distance and convert to similarity
                dist = torch.cdist(embeddings1, embeddings2, p=2)
                similarity = 1.0 / (1.0 + dist)
            else:
                raise ValueError(f"Unsupported similarity type: {similarity_type}")
                
            return similarity.cpu()
            
        except Exception as e:
            self.logger.error(f"Similarity computation failed: {e}")
            raise DiscoMusicaError(f"Similarity computation failed: {e}")
            
    async def find_nearest_neighbors(
        self,
        query_embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
        k: int = 5,
        similarity_type: str = "cosine"
    ) -> Dict[str, Any]:
        """Find nearest neighbors for query embeddings.
        
        Args:
            query_embeddings: Query embeddings.
            reference_embeddings: Reference embeddings.
            k: Number of nearest neighbors to find.
            similarity_type: Type of similarity to compute.
            
        Returns:
            Nearest neighbors and distances.
        """
        try:
            # Compute similarity matrix
            similarity = await self.compute_similarity(
                query_embeddings,
                reference_embeddings,
                similarity_type
            )
            
            # Find top-k nearest neighbors
            values, indices = torch.topk(similarity, k=k, dim=-1)
            
            return {
                "indices": indices.cpu().numpy(),
                "similarities": values.cpu().numpy(),
                "metadata": {
                    "k": k,
                    "similarity_type": similarity_type
                }
            }
            
        except Exception as e:
            self.logger.error(f"Nearest neighbor search failed: {e}")
            raise DiscoMusicaError(f"Nearest neighbor search failed: {e}")
            
    async def cluster_embeddings(
        self,
        embeddings: torch.Tensor,
        n_clusters: int = 10,
        algorithm: str = "kmeans"
    ) -> Dict[str, Any]:
        """Cluster embeddings.
        
        Args:
            embeddings: Input embeddings.
            n_clusters: Number of clusters.
            algorithm: Clustering algorithm ("kmeans", "dbscan").
            
        Returns:
            Cluster assignments and metadata.
        """
        try:
            # Move embeddings to device
            embeddings = embeddings.to(self.device)
            
            # Cluster embeddings
            if algorithm == "kmeans":
                # Implement k-means clustering
                centroids = embeddings[torch.randperm(len(embeddings))[:n_clusters]]
                assignments = torch.zeros(len(embeddings), dtype=torch.long)
                
                for _ in range(100):  # Max iterations
                    # Assign points to nearest centroid
                    distances = torch.cdist(embeddings, centroids)
                    new_assignments = torch.argmin(distances, dim=-1)
                    
                    # Check convergence
                    if torch.all(new_assignments == assignments):
                        break
                        
                    assignments = new_assignments
                    
                    # Update centroids
                    for i in range(n_clusters):
                        mask = assignments == i
                        if torch.any(mask):
                            centroids[i] = embeddings[mask].mean(dim=0)
                            
            elif algorithm == "dbscan":
                # Implement DBSCAN clustering
                # Note: This is a simplified version
                distances = torch.cdist(embeddings, embeddings)
                assignments = torch.zeros(len(embeddings), dtype=torch.long)
                current_cluster = 1
                
                for i in range(len(embeddings)):
                    if assignments[i] == 0:
                        # Find neighbors
                        neighbors = torch.where(distances[i] < 0.5)[0]
                        if len(neighbors) >= 5:  # Min points
                            assignments[neighbors] = current_cluster
                            current_cluster += 1
                            
            else:
                raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
                
            return {
                "assignments": assignments.cpu().numpy(),
                "metadata": {
                    "n_clusters": n_clusters,
                    "algorithm": algorithm
                }
            }
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            raise DiscoMusicaError(f"Clustering failed: {e}")
            
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
    def load(cls, path: str) -> "VectorEmbeddingModel":
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
            config = VectorEmbeddingConfig(**checkpoint["config"])
            
            # Create model
            model = cls(config)
            
            # Load state dict
            model.model.load_state_dict(checkpoint["model_state_dict"])
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise DiscoMusicaError(f"Failed to load model: {e}") 