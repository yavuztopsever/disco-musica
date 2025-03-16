"""PostgreSQL storage for metadata."""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import asyncpg
from asyncpg.exceptions import PostgresError

from ...core.exceptions.base_exceptions import ProcessingError


class PostgresStorage:
    """PostgreSQL storage for metadata."""
    
    def __init__(
        self,
        uri: str,
        database: str = "disco_musica"
    ):
        """Initialize PostgreSQL storage.
        
        Args:
            uri: PostgreSQL connection URI.
            database: Database name.
        """
        self.uri = uri
        self.database = database
        self.pool = None
        self.logger = logging.getLogger(__name__)
        
    async def connect(self):
        """Connect to database."""
        try:
            self.pool = await asyncpg.create_pool(
                self.uri,
                database=self.database,
                min_size=1,
                max_size=10
            )
            await self._create_tables()
        except PostgresError as e:
            self.logger.error(f"PostgreSQL error: {str(e)}")
            raise ProcessingError(
                f"Failed to connect to database: {str(e)}"
            )
            
    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()
            
    async def _create_tables(self):
        """Create required tables."""
        async with self.pool.acquire() as conn:
            # Create resource metadata table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS resource_metadata (
                    resource_id TEXT PRIMARY KEY,
                    resource_type TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create generation metadata table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS generation_metadata (
                    output_id TEXT PRIMARY KEY,
                    generation_type TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_resource_type 
                ON resource_metadata(resource_type)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_generation_type 
                ON generation_metadata(generation_type)
            """)
            
    async def insert_resource_metadata(
        self,
        resource_id: str,
        resource_type: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Insert resource metadata.
        
        Args:
            resource_id: Resource ID.
            resource_type: Resource type.
            metadata: Resource metadata.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO resource_metadata (
                        resource_id, resource_type, metadata
                    ) VALUES ($1, $2, $3)
                    ON CONFLICT (resource_id) DO UPDATE SET
                        resource_type = EXCLUDED.resource_type,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, resource_id, resource_type, metadata)
        except PostgresError as e:
            self.logger.error(f"PostgreSQL error: {str(e)}")
            raise ProcessingError(
                f"Failed to insert resource metadata: {str(e)}"
            )
            
    async def update_resource_metadata(
        self,
        resource_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Update resource metadata.
        
        Args:
            resource_id: Resource ID.
            metadata: Updated metadata.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE resource_metadata
                    SET metadata = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE resource_id = $2
                """, metadata, resource_id)
        except PostgresError as e:
            self.logger.error(f"PostgreSQL error: {str(e)}")
            raise ProcessingError(
                f"Failed to update resource metadata: {str(e)}"
            )
            
    async def delete_resource_metadata(
        self,
        resource_id: str
    ) -> None:
        """Delete resource metadata.
        
        Args:
            resource_id: Resource ID.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    DELETE FROM resource_metadata
                    WHERE resource_id = $1
                """, resource_id)
        except PostgresError as e:
            self.logger.error(f"PostgreSQL error: {str(e)}")
            raise ProcessingError(
                f"Failed to delete resource metadata: {str(e)}"
            )
            
    async def get_resource_metadata(
        self,
        resource_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get resource metadata.
        
        Args:
            resource_id: Resource ID.
            
        Returns:
            Resource metadata, or None if not found.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT metadata
                    FROM resource_metadata
                    WHERE resource_id = $1
                """, resource_id)
                return row["metadata"] if row else None
        except PostgresError as e:
            self.logger.error(f"PostgreSQL error: {str(e)}")
            raise ProcessingError(
                f"Failed to get resource metadata: {str(e)}"
            )
            
    async def insert_generation_metadata(
        self,
        output_id: str,
        generation_type: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Insert generation metadata.
        
        Args:
            output_id: Output ID.
            generation_type: Generation type.
            metadata: Generation metadata.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO generation_metadata (
                        output_id, generation_type, metadata
                    ) VALUES ($1, $2, $3)
                    ON CONFLICT (output_id) DO UPDATE SET
                        generation_type = EXCLUDED.generation_type,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, output_id, generation_type, metadata)
        except PostgresError as e:
            self.logger.error(f"PostgreSQL error: {str(e)}")
            raise ProcessingError(
                f"Failed to insert generation metadata: {str(e)}"
            )
            
    async def update_generation_metadata(
        self,
        output_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Update generation metadata.
        
        Args:
            output_id: Output ID.
            metadata: Updated metadata.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE generation_metadata
                    SET metadata = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE output_id = $2
                """, metadata, output_id)
        except PostgresError as e:
            self.logger.error(f"PostgreSQL error: {str(e)}")
            raise ProcessingError(
                f"Failed to update generation metadata: {str(e)}"
            )
            
    async def delete_generation_metadata(
        self,
        output_id: str
    ) -> None:
        """Delete generation metadata.
        
        Args:
            output_id: Output ID.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    DELETE FROM generation_metadata
                    WHERE output_id = $1
                """, output_id)
        except PostgresError as e:
            self.logger.error(f"PostgreSQL error: {str(e)}")
            raise ProcessingError(
                f"Failed to delete generation metadata: {str(e)}"
            )
            
    async def get_generation_metadata(
        self,
        output_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get generation metadata.
        
        Args:
            output_id: Output ID.
            
        Returns:
            Generation metadata, or None if not found.
            
        Raises:
            ProcessingError: If database error occurs.
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT metadata
                    FROM generation_metadata
                    WHERE output_id = $1
                """, output_id)
                return row["metadata"] if row else None
        except PostgresError as e:
            self.logger.error(f"PostgreSQL error: {str(e)}")
            raise ProcessingError(
                f"Failed to get generation metadata: {str(e)}"
            ) 