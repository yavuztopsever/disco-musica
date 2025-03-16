"""Vector search interface for handling similarity searches."""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
from pydantic import BaseModel

from ...core.exceptions import DiscoMusicaError
from ...core.resources import BaseResource
from ...data.storage.vector_storage import VectorStorage
from ...models.vector.vector_embedding_model import VectorEmbeddingModel

logger = logging.getLogger(__name__)

class SearchQuery(BaseModel):
    """Model for search queries."""
    query: Union[str, List[float], np.ndarray]
    query_type: str = "text"  # text, embedding, audio, midi
    top_k: int = 10
    metadata_filter: Optional[Dict[str, Any]] = None
    similarity_type: str = "cosine"

class SearchResult(BaseModel):
    """Model for search results."""
    resource_id: str
    score: float
    metadata: Dict[str, Any]

class VectorSearchInterface:
    """Interface for handling similarity searches."""
    
    def __init__(
        self,
        vector_storage: VectorStorage,
        embedding_model: VectorEmbeddingModel
    ):
        """Initialize the vector search interface.
        
        Args:
            vector_storage: Vector storage service.
            embedding_model: Vector embedding model.
        """
        self.vector_storage = vector_storage
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)
        
    async def search(
        self,
        query: SearchQuery
    ) -> List[SearchResult]:
        """Search for similar items.
        
        Args:
            query: Search query.
            
        Returns:
            List of search results.
        """
        try:
            # Get query embedding
            query_embedding = await self._get_query_embedding(query)
            
            # Search vector storage
            results = await self.vector_storage.search_similar(
                query_embedding=query_embedding,
                top_k=query.top_k,
                metadata_filter=query.metadata_filter
            )
            
            # Convert to search results
            search_results = [
                SearchResult(
                    resource_id=result["resource_id"],
                    score=result["score"],
                    metadata=result["metadata"]
                )
                for result in results
            ]
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise DiscoMusicaError(f"Search failed: {e}")
            
    async def batch_search(
        self,
        queries: List[SearchQuery]
    ) -> List[List[SearchResult]]:
        """Search for similar items for multiple queries.
        
        Args:
            queries: List of search queries.
            
        Returns:
            List of search results for each query.
        """
        try:
            # Get query embeddings
            query_embeddings = []
            for query in queries:
                embedding = await self._get_query_embedding(query)
                query_embeddings.append(embedding)
                
            # Search vector storage
            batch_results = await self.vector_storage.batch_search_similar(
                query_embeddings=query_embeddings,
                top_k=max(query.top_k for query in queries),
                metadata_filter=queries[0].metadata_filter  # Use first query's filter
            )
            
            # Convert to search results
            search_results = []
            for results in batch_results:
                query_results = [
                    SearchResult(
                        resource_id=result["resource_id"],
                        score=result["score"],
                        metadata=result["metadata"]
                    )
                    for result in results
                ]
                search_results.append(query_results)
                
            return search_results
            
        except Exception as e:
            self.logger.error(f"Batch search failed: {e}")
            raise DiscoMusicaError(f"Batch search failed: {e}")
            
    async def _get_query_embedding(
        self,
        query: SearchQuery
    ) -> np.ndarray:
        """Get embedding for query.
        
        Args:
            query: Search query.
            
        Returns:
            Query embedding.
        """
        if query.query_type == "embedding":
            # Query is already an embedding
            if isinstance(query.query, np.ndarray):
                return query.query
            elif isinstance(query.query, list):
                return np.array(query.query)
            else:
                raise ValueError(f"Invalid embedding type: {type(query.query)}")
                
        elif query.query_type == "text":
            # Generate text embedding
            if not isinstance(query.query, str):
                raise ValueError(f"Invalid text query type: {type(query.query)}")
                
            # Convert text to token IDs (implementation depends on tokenizer)
            token_ids = self._tokenize_text(query.query)
            
            # Generate embedding
            outputs = await self.embedding_model.encode(
                input_ids=token_ids
            )
            
            return outputs["embeddings"]
            
        elif query.query_type == "audio":
            # Generate audio embedding
            if not isinstance(query.query, (np.ndarray, list)):
                raise ValueError(f"Invalid audio query type: {type(query.query)}")
                
            # Process audio and generate embedding
            audio_features = self._extract_audio_features(query.query)
            
            # Generate embedding
            outputs = await self.embedding_model.encode(
                input_ids=audio_features
            )
            
            return outputs["embeddings"]
            
        elif query.query_type == "midi":
            # Generate MIDI embedding
            if not isinstance(query.query, (np.ndarray, list)):
                raise ValueError(f"Invalid MIDI query type: {type(query.query)}")
                
            # Process MIDI and generate embedding
            midi_features = self._extract_midi_features(query.query)
            
            # Generate embedding
            outputs = await self.embedding_model.encode(
                input_ids=midi_features
            )
            
            return outputs["embeddings"]
            
        else:
            raise ValueError(f"Unsupported query type: {query.query_type}")
            
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text into token IDs.
        
        Args:
            text: Input text.
            
        Returns:
            Token IDs tensor.
        """
        # TODO: Implement text tokenization
        # This would use the appropriate tokenizer for the embedding model
        pass
        
    def _extract_audio_features(
        self,
        audio: Union[np.ndarray, List[float]]
    ) -> torch.Tensor:
        """Extract features from audio data.
        
        Args:
            audio: Audio data.
            
        Returns:
            Audio features tensor.
        """
        # TODO: Implement audio feature extraction
        # This would use appropriate audio processing libraries
        pass
        
    def _extract_midi_features(
        self,
        midi: Union[np.ndarray, List[float]]
    ) -> torch.Tensor:
        """Extract features from MIDI data.
        
        Args:
            midi: MIDI data.
            
        Returns:
            MIDI features tensor.
        """
        # TODO: Implement MIDI feature extraction
        # This would use appropriate MIDI processing libraries
        pass
        
    async def cluster_search_results(
        self,
        results: List[SearchResult],
        n_clusters: int = 5,
        algorithm: str = "kmeans"
    ) -> Dict[str, Any]:
        """Cluster search results.
        
        Args:
            results: List of search results.
            n_clusters: Number of clusters.
            algorithm: Clustering algorithm.
            
        Returns:
            Clustering results.
        """
        try:
            # Get embeddings for results
            embeddings = []
            for result in results:
                embedding = await self.vector_storage.get_embedding(result.resource_id)
                if embedding is not None:
                    embeddings.append(embedding.embedding)
                    
            if not embeddings:
                return {
                    "clusters": [],
                    "metadata": {
                        "n_clusters": 0,
                        "algorithm": algorithm
                    }
                }
                
            # Convert to tensor
            embeddings_tensor = torch.tensor(embeddings)
            
            # Cluster embeddings
            cluster_results = await self.embedding_model.cluster_embeddings(
                embeddings=embeddings_tensor,
                n_clusters=min(n_clusters, len(embeddings)),
                algorithm=algorithm
            )
            
            # Assign results to clusters
            clusters = []
            for i in range(cluster_results["metadata"]["n_clusters"]):
                cluster_mask = cluster_results["assignments"] == i
                cluster_results = [
                    result for j, result in enumerate(results)
                    if cluster_mask[j]
                ]
                clusters.append(cluster_results)
                
            return {
                "clusters": clusters,
                "metadata": cluster_results["metadata"]
            }
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            raise DiscoMusicaError(f"Clustering failed: {e}")
            
    async def analyze_search_results(
        self,
        results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Analyze search results.
        
        Args:
            results: List of search results.
            
        Returns:
            Analysis results.
        """
        try:
            # Calculate basic statistics
            scores = [result.score for result in results]
            
            stats = {
                "count": len(results),
                "mean_score": float(np.mean(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "std_score": float(np.std(scores))
            }
            
            # Analyze metadata distribution
            metadata_stats = {}
            for result in results:
                for key, value in result.metadata.items():
                    if key not in metadata_stats:
                        metadata_stats[key] = {}
                        
                    if isinstance(value, (str, int, float, bool)):
                        if value not in metadata_stats[key]:
                            metadata_stats[key][value] = 0
                        metadata_stats[key][value] += 1
                        
            return {
                "statistics": stats,
                "metadata_distribution": metadata_stats
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise DiscoMusicaError(f"Analysis failed: {e}")
            
    async def filter_search_results(
        self,
        results: List[SearchResult],
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Filter search results.
        
        Args:
            results: List of search results.
            min_score: Minimum similarity score.
            max_score: Maximum similarity score.
            metadata_filter: Metadata filter.
            
        Returns:
            Filtered search results.
        """
        try:
            filtered_results = []
            
            for result in results:
                # Apply score filters
                if min_score is not None and result.score < min_score:
                    continue
                if max_score is not None and result.score > max_score:
                    continue
                    
                # Apply metadata filter
                if metadata_filter:
                    if not all(
                        result.metadata.get(k) == v
                        for k, v in metadata_filter.items()
                    ):
                        continue
                        
                filtered_results.append(result)
                
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Filtering failed: {e}")
            raise DiscoMusicaError(f"Filtering failed: {e}")
            
    async def rank_search_results(
        self,
        results: List[SearchResult],
        ranking_criteria: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """Rank search results using custom criteria.
        
        Args:
            results: List of search results.
            ranking_criteria: Dictionary of criteria and weights.
            
        Returns:
            Ranked search results.
        """
        try:
            if not ranking_criteria:
                # Use default ranking by similarity score
                return sorted(
                    results,
                    key=lambda x: x.score,
                    reverse=True
                )
                
            # Calculate weighted scores
            ranked_results = []
            for result in results:
                weighted_score = 0.0
                
                # Add similarity score component
                if "similarity" in ranking_criteria:
                    weighted_score += ranking_criteria["similarity"] * result.score
                    
                # Add metadata-based components
                for key, weight in ranking_criteria.items():
                    if key != "similarity" and key in result.metadata:
                        value = result.metadata[key]
                        if isinstance(value, (int, float)):
                            weighted_score += weight * value
                            
                ranked_results.append((result, weighted_score))
                
            # Sort by weighted score
            ranked_results.sort(key=lambda x: x[1], reverse=True)
            
            return [result for result, _ in ranked_results]
            
        except Exception as e:
            self.logger.error(f"Ranking failed: {e}")
            raise DiscoMusicaError(f"Ranking failed: {e}") 