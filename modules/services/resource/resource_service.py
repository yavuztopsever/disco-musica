"""Service for managing resources."""
import logging
from typing import Dict, Any, List, Optional, Type
from datetime import datetime

from ...core.exceptions.base_exceptions import ResourceNotFoundError
from ...core.resources.base_resources import Resource, ProjectResource, TrackResource, ModelResource
from ...data.storage.mongo_storage import MongoStorage
from ...data.storage.postgres_storage import PostgresStorage


class ResourceService:
    """Service for managing resources."""
    
    def __init__(
        self,
        mongo_storage: MongoStorage,
        postgres_storage: PostgresStorage
    ):
        """Initialize the resource service.
        
        Args:
            mongo_storage: MongoDB storage for resources.
            postgres_storage: PostgreSQL storage for metadata.
        """
        self.mongo_storage = mongo_storage
        self.postgres_storage = postgres_storage
        self.logger = logging.getLogger(__name__)
        
        # Map resource types to classes
        self._resource_classes = {
            "project": ProjectResource,
            "track": TrackResource,
            "model": ModelResource
        }
        
    async def get_resource(
        self,
        resource_id: str,
        resource_type: Optional[str] = None
    ) -> Resource:
        """Get a resource by ID.
        
        Args:
            resource_id: Resource ID.
            resource_type: Optional resource type.
            
        Returns:
            Resource instance.
            
        Raises:
            ResourceNotFoundError: If resource not found.
        """
        # Get resource data from MongoDB
        resource_data = await self.mongo_storage.get_document(
            collection="resources",
            document_id=resource_id
        )
        
        if not resource_data:
            raise ResourceNotFoundError(
                f"Resource {resource_id} not found"
            )
            
        # Get resource type
        if not resource_type:
            resource_type = resource_data.get("resource_type")
            if not resource_type:
                raise ResourceNotFoundError(
                    f"Resource {resource_id} has no type"
                )
                
        # Get resource class
        resource_class = self._get_resource_class(resource_type)
        
        # Create resource instance
        resource = resource_class(**resource_data)
        
        return resource
        
    async def create_resource(
        self,
        resource_type: str,
        data: Dict[str, Any],
        parent_id: Optional[str] = None
    ) -> Resource:
        """Create a new resource.
        
        Args:
            resource_type: Type of resource to create.
            data: Resource data.
            parent_id: Optional parent resource ID.
            
        Returns:
            Created resource instance.
        """
        # Get resource class
        resource_class = self._get_resource_class(resource_type)
        
        # Create resource instance
        resource = resource_class(
            resource_type=resource_type,
            **data
        )
        
        # Set parent if provided
        if parent_id:
            resource.parent_resources.append(parent_id)
            
        # Save to MongoDB
        await self.mongo_storage.insert_document(
            collection="resources",
            document=resource.dict()
        )
        
        # Save metadata to PostgreSQL
        await self.postgres_storage.insert_resource_metadata(
            resource_id=resource.resource_id,
            resource_type=resource_type,
            metadata=data
        )
        
        return resource
        
    async def update_resource(
        self,
        resource_id: str,
        data: Dict[str, Any]
    ) -> Resource:
        """Update a resource.
        
        Args:
            resource_id: Resource ID.
            data: Updated resource data.
            
        Returns:
            Updated resource instance.
            
        Raises:
            ResourceNotFoundError: If resource not found.
        """
        # Get existing resource
        resource = await self.get_resource(resource_id)
        
        # Update resource data
        for key, value in data.items():
            setattr(resource, key, value)
            
        # Update timestamps
        resource.modification_timestamp = datetime.utcnow()
        resource.version += 1
        
        # Save to MongoDB
        await self.mongo_storage.update_document(
            collection="resources",
            document_id=resource_id,
            document=resource.dict()
        )
        
        # Update metadata in PostgreSQL
        await self.postgres_storage.update_resource_metadata(
            resource_id=resource_id,
            metadata=data
        )
        
        return resource
        
    async def delete_resource(
        self,
        resource_id: str
    ) -> None:
        """Delete a resource.
        
        Args:
            resource_id: Resource ID.
            
        Raises:
            ResourceNotFoundError: If resource not found.
        """
        # Check if resource exists
        resource = await self.get_resource(resource_id)
        
        # Delete from MongoDB
        await self.mongo_storage.delete_document(
            collection="resources",
            document_id=resource_id
        )
        
        # Delete metadata from PostgreSQL
        await self.postgres_storage.delete_resource_metadata(
            resource_id=resource_id
        )
        
    async def list_resources(
        self,
        resource_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        **filters
    ) -> List[Resource]:
        """List resources with optional filters.
        
        Args:
            resource_type: Optional resource type filter.
            parent_id: Optional parent resource ID filter.
            **filters: Additional filters.
            
        Returns:
            List of resource instances.
        """
        # Build query
        query = {}
        if resource_type:
            query["resource_type"] = resource_type
        if parent_id:
            query["parent_resources"] = parent_id
        query.update(filters)
        
        # Get resources from MongoDB
        resource_data = await self.mongo_storage.find_documents(
            collection="resources",
            query=query
        )
        
        # Create resource instances
        resources = []
        for data in resource_data:
            resource_type = data.get("resource_type")
            if resource_type:
                resource_class = self._get_resource_class(resource_type)
                resource = resource_class(**data)
                resources.append(resource)
                
        return resources
        
    def _get_resource_class(self, resource_type: str) -> Type[Resource]:
        """Get resource class for type.
        
        Args:
            resource_type: Type of resource.
            
        Returns:
            Resource class.
            
        Raises:
            ValueError: If resource type not found.
        """
        if resource_type not in self._resource_classes:
            raise ValueError(
                f"Unknown resource type: {resource_type}"
            )
            
        return self._resource_classes[resource_type] 