"""Vector storage service for handling vector embeddings and similarity search."""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
from pydantic import BaseModel

from ...core.exceptions import DiscoMusicaError
from ...core.resources import BaseResource

logger = logging.getLogger(__name__)

class VectorEmbedding(BaseModel):
    """Model for vector embeddings."""
    resource_id: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: str

class VectorStorage:
    """Service for managing vector embeddings and similarity search."""
    
    def __init__(self, dimension: int = 768):
        """Initialize the vector storage.
        
        Args:
            dimension: Dimension of the vector embeddings.
        """
        self.dimension = dimension
        self.embeddings: Dict[str, VectorEmbedding] = {}
        self.logger = logging.getLogger(__name__)
        
    async def store_embedding(
        self,
        resource_id: str,
        embedding: Union[List[float], np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a vector embedding.
        
        Args:
            resource_id: ID of the resource.
            embedding: Vector embedding.
            metadata: Optional metadata.
            
        Returns:
            ID of the stored embedding.
        """
        # Convert embedding to list if numpy array
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
            
        # Validate dimension
        if len(embedding) != self.dimension:
            raise ValueError(f"Embedding dimension {len(embedding)} does not match expected dimension {self.dimension}")
            
        # Create embedding object
        vector_embedding = VectorEmbedding(
            resource_id=resource_id,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=str(datetime.utcnow())
        )
        
        # Store embedding
        self.embeddings[resource_id] = vector_embedding
        
        return resource_id
        
    async def get_embedding(self, resource_id: str) -> Optional[VectorEmbedding]:
        """Get a vector embedding by ID.
        
        Args:
            resource_id: ID of the resource.
            
        Returns:
            Vector embedding if found, None otherwise.
        """
        return self.embeddings.get(resource_id)
        
    async def delete_embedding(self, resource_id: str) -> bool:
        """Delete a vector embedding.
        
        Args:
            resource_id: ID of the resource.
            
        Returns:
            True if deleted, False if not found.
        """
        if resource_id in self.embeddings:
            del self.embeddings[resource_id]
            return True
        return False
        
    async def search_similar(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Query vector embedding.
            top_k: Number of results to return.
            metadata_filter: Optional metadata filter.
            
        Returns:
            List of similar embeddings with scores.
        """
        # Convert query to numpy array
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
            
        # Validate dimension
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} does not match expected dimension {self.dimension}")
            
        # Calculate similarities
        similarities = []
        for resource_id, embedding in self.embeddings.items():
            # Apply metadata filter if provided
            if metadata_filter:
                if not all(embedding.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
                    
            # Calculate cosine similarity
            embedding_array = np.array(embedding.embedding)
            similarity = np.dot(query_embedding, embedding_array) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding_array)
            )
            
            similarities.append({
                "resource_id": resource_id,
                "score": float(similarity),
                "metadata": embedding.metadata
            })
            
        # Sort by similarity score
        similarities.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top k results
        return similarities[:top_k]
        
    async def batch_store_embeddings(
        self,
        embeddings: List[Dict[str, Any]]
    ) -> List[str]:
        """Store multiple embeddings in batch.
        
        Args:
            embeddings: List of embedding dictionaries with resource_id, embedding, and metadata.
            
        Returns:
            List of stored embedding IDs.
        """
        stored_ids = []
        for embedding_data in embeddings:
            try:
                resource_id = await self.store_embedding(
                    resource_id=embedding_data["resource_id"],
                    embedding=embedding_data["embedding"],
                    metadata=embedding_data.get("metadata")
                )
                stored_ids.append(resource_id)
            except Exception as e:
                self.logger.error(f"Error storing embedding {embedding_data['resource_id']}: {e}")
                
        return stored_ids
        
    async def batch_search_similar(
        self,
        query_embeddings: List[Union[List[float], np.ndarray]],
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """Search for similar embeddings for multiple queries.
        
        Args:
            query_embeddings: List of query vector embeddings.
            top_k: Number of results to return per query.
            metadata_filter: Optional metadata filter.
            
        Returns:
            List of results for each query.
        """
        results = []
        for query_embedding in query_embeddings:
            try:
                query_results = await self.search_similar(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    metadata_filter=metadata_filter
                )
                results.append(query_results)
            except Exception as e:
                self.logger.error(f"Error searching with query embedding: {e}")
                results.append([])
                
        return results 