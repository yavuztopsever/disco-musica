"""Service for managing generation outputs."""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...core.exceptions.base_exceptions import ProcessingError
from ...core.resources.base_resources import Resource
from ...data.storage.mongo_storage import MongoStorage
from ...data.storage.postgres_storage import PostgresStorage
from ...utils.metrics.prometheus_metrics import GENERATION_SIZES


class OutputService:
    """Service for managing generation outputs."""
    
    def __init__(
        self,
        mongo_storage: MongoStorage,
        postgres_storage: PostgresStorage
    ):
        """Initialize the output service.
        
        Args:
            mongo_storage: MongoDB storage for outputs.
            postgres_storage: PostgreSQL storage for metadata.
        """
        self.mongo_storage = mongo_storage
        self.postgres_storage = postgres_storage
        self.logger = logging.getLogger(__name__)
        
    async def save_generation(
        self,
        outputs: Dict[str, Any],
        generation_type: str,
        source_text: Optional[str] = None,
        source_midi: Optional[bytes] = None,
        source_audio: Optional[bytes] = None,
        project_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save generation outputs.
        
        Args:
            outputs: Generation outputs.
            generation_type: Type of generation.
            source_text: Optional source text.
            source_midi: Optional source MIDI data.
            source_audio: Optional source audio data.
            project_id: Optional project ID.
            config: Optional generation configuration.
            
        Returns:
            Output ID.
        """
        self.logger.info(f"Saving {generation_type} generation outputs...")
        
        # Create output document
        output_doc = {
            "generation_type": generation_type,
            "timestamp": datetime.utcnow(),
            "outputs": outputs,
            "source": {
                "text": source_text,
                "midi": source_midi,
                "audio": source_audio
            },
            "project_id": project_id,
            "config": config or {}
        }
        
        # Save to MongoDB
        output_id = await self.mongo_storage.insert_document(
            collection="generations",
            document=output_doc
        )
        
        # Track output sizes
        for content_type, content in outputs.items():
            if isinstance(content, (bytes, bytearray)):
                GENERATION_SIZES.labels(content_type=content_type).observe(len(content))
                
        # Save metadata to PostgreSQL
        await self.postgres_storage.insert_generation_metadata(
            output_id=output_id,
            generation_type=generation_type,
            metadata={
                "source_text": source_text,
                "project_id": project_id,
                "config": config
            }
        )
        
        return output_id
        
    async def get_generation(
        self,
        output_id: str
    ) -> Dict[str, Any]:
        """Get generation outputs by ID.
        
        Args:
            output_id: Output ID.
            
        Returns:
            Generation outputs.
            
        Raises:
            ProcessingError: If output not found.
        """
        # Get output from MongoDB
        output = await self.mongo_storage.get_document(
            collection="generations",
            document_id=output_id
        )
        
        if not output:
            raise ProcessingError(
                f"Generation output {output_id} not found"
            )
            
        return output
        
    async def list_generations(
        self,
        generation_type: Optional[str] = None,
        project_id: Optional[str] = None,
        **filters
    ) -> List[Dict[str, Any]]:
        """List generation outputs with optional filters.
        
        Args:
            generation_type: Optional generation type filter.
            project_id: Optional project ID filter.
            **filters: Additional filters.
            
        Returns:
            List of generation outputs.
        """
        # Build query
        query = {}
        if generation_type:
            query["generation_type"] = generation_type
        if project_id:
            query["project_id"] = project_id
        query.update(filters)
        
        # Get outputs from MongoDB
        outputs = await self.mongo_storage.find_documents(
            collection="generations",
            query=query
        )
        
        return outputs
        
    async def delete_generation(
        self,
        output_id: str
    ) -> None:
        """Delete generation outputs.
        
        Args:
            output_id: Output ID.
            
        Raises:
            ProcessingError: If output not found.
        """
        # Check if output exists
        output = await self.get_generation(output_id)
        
        # Delete from MongoDB
        await self.mongo_storage.delete_document(
            collection="generations",
            document_id=output_id
        )
        
        # Delete metadata from PostgreSQL
        await self.postgres_storage.delete_generation_metadata(
            output_id=output_id
        ) 