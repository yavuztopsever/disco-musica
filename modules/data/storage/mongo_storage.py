"""MongoDB storage for resources and outputs."""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import motor.motor_asyncio
from bson import ObjectId
from pymongo.errors import PyMongoError

from ...core.exceptions.base_exceptions import ProcessingError


class MongoStorage:
    """MongoDB storage for resources and outputs."""
    
    def __init__(
        self,
        uri: str,
        database: str = "disco_musica"
    ):
        """Initialize MongoDB storage.
        
        Args:
            uri: MongoDB connection URI.
            database: Database name.
        """
        self.client = motor.motor_asyncio.AsyncIOMotorClient(uri)
        self.db = self.client[database]
        self.logger = logging.getLogger(__name__)
        
    async def get_document(
        self,
        collection: str,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a document by ID.
        
        Args:
            collection: Collection name.
            document_id: Document ID.
            
        Returns:
            Document data, or None if not found.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            document = await self.db[collection].find_one(
                {"_id": ObjectId(document_id)}
            )
            return document
        except PyMongoError as e:
            self.logger.error(f"MongoDB error: {str(e)}")
            raise ProcessingError(
                f"Failed to get document: {str(e)}"
            )
            
    async def insert_document(
        self,
        collection: str,
        document: Dict[str, Any]
    ) -> str:
        """Insert a document.
        
        Args:
            collection: Collection name.
            document: Document data.
            
        Returns:
            Document ID.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            # Add timestamps
            document["created_at"] = datetime.utcnow()
            document["updated_at"] = datetime.utcnow()
            
            # Insert document
            result = await self.db[collection].insert_one(document)
            return str(result.inserted_id)
        except PyMongoError as e:
            self.logger.error(f"MongoDB error: {str(e)}")
            raise ProcessingError(
                f"Failed to insert document: {str(e)}"
            )
            
    async def update_document(
        self,
        collection: str,
        document_id: str,
        document: Dict[str, Any]
    ) -> None:
        """Update a document.
        
        Args:
            collection: Collection name.
            document_id: Document ID.
            document: Updated document data.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            # Update timestamp
            document["updated_at"] = datetime.utcnow()
            
            # Update document
            await self.db[collection].update_one(
                {"_id": ObjectId(document_id)},
                {"$set": document}
            )
        except PyMongoError as e:
            self.logger.error(f"MongoDB error: {str(e)}")
            raise ProcessingError(
                f"Failed to update document: {str(e)}"
            )
            
    async def delete_document(
        self,
        collection: str,
        document_id: str
    ) -> None:
        """Delete a document.
        
        Args:
            collection: Collection name.
            document_id: Document ID.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            await self.db[collection].delete_one(
                {"_id": ObjectId(document_id)}
            )
        except PyMongoError as e:
            self.logger.error(f"MongoDB error: {str(e)}")
            raise ProcessingError(
                f"Failed to delete document: {str(e)}"
            )
            
    async def find_documents(
        self,
        collection: str,
        query: Dict[str, Any],
        sort: Optional[List[tuple]] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find documents matching query.
        
        Args:
            collection: Collection name.
            query: Query filter.
            sort: Optional sort criteria.
            limit: Optional limit.
            skip: Optional skip.
            
        Returns:
            List of matching documents.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            cursor = self.db[collection].find(query)
            
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
                
            documents = await cursor.to_list(length=None)
            return documents
        except PyMongoError as e:
            self.logger.error(f"MongoDB error: {str(e)}")
            raise ProcessingError(
                f"Failed to find documents: {str(e)}"
            )
            
    async def count_documents(
        self,
        collection: str,
        query: Dict[str, Any]
    ) -> int:
        """Count documents matching query.
        
        Args:
            collection: Collection name.
            query: Query filter.
            
        Returns:
            Number of matching documents.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            count = await self.db[collection].count_documents(query)
            return count
        except PyMongoError as e:
            self.logger.error(f"MongoDB error: {str(e)}")
            raise ProcessingError(
                f"Failed to count documents: {str(e)}"
            )
            
    async def create_index(
        self,
        collection: str,
        keys: List[tuple],
        unique: bool = False
    ) -> None:
        """Create an index.
        
        Args:
            collection: Collection name.
            keys: Index keys.
            unique: Whether index is unique.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            await self.db[collection].create_index(
                keys,
                unique=unique
            )
        except PyMongoError as e:
            self.logger.error(f"MongoDB error: {str(e)}")
            raise ProcessingError(
                f"Failed to create index: {str(e)}"
            ) 