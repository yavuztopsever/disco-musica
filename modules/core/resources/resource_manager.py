"""Resource Manager for handling system resources."""

import json
from typing import Dict, List, Optional, Union, Any, Type
from pathlib import Path
import logging
from datetime import datetime
import uuid
import sqlite3
from contextlib import contextmanager

from .base_resource import BaseResource
from .model_resource import ModelResource
from .track_resource import TrackResource
from .generation_resource import GenerationResource
from ..exceptions.base_exceptions import (
    ProcessingError,
    ValidationError,
    ResourceNotFoundError
)


class ResourceManager:
    """Manager for handling system resources.
    
    This class provides functionality for managing different types of resources
    including models, tracks, and generations. It handles resource creation,
    retrieval, updates, and deletion with persistent storage.
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """Initialize the resource manager.
        
        Args:
            base_path: Base path for resource storage.
        """
        self.base_path = Path(base_path)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Resource type mappings
        self.resource_types = {
            "model": ModelResource,
            "track": TrackResource,
            "generation": GenerationResource
        }
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Initialize database
        self._init_database()
        
    def _ensure_directories(self) -> None:
        """Ensure required resource directories exist."""
        directories = [
            self.base_path,
            self.base_path / "models",
            self.base_path / "tracks",
            self.base_path / "generations",
            self.base_path / "db"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def _init_database(self) -> None:
        """Initialize SQLite database for resource metadata."""
        db_path = self.base_path / "db" / "resources.db"
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables for each resource type
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resources (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    project_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            # Create indices
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_type ON resources(type)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_project ON resources(project_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_status ON resources(status)"
            )
            
    @contextmanager
    def _get_db_connection(self):
        """Get database connection context manager."""
        db_path = self.base_path / "db" / "resources.db"
        conn = sqlite3.connect(str(db_path))
        try:
            yield conn
        finally:
            conn.close()
            
    def create_resource(
        self,
        resource_type: str,
        name: str,
        project_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new resource.
        
        Args:
            resource_type: Type of resource.
            name: Resource name.
            project_id: Optional project ID.
            **kwargs: Additional resource attributes.
            
        Returns:
            Created resource information.
            
        Raises:
            ValidationError: If resource parameters are invalid.
            ProcessingError: If creation fails.
        """
        try:
            # Validate resource type
            if resource_type not in self.resource_types:
                raise ValidationError(f"Invalid resource type: {resource_type}")
                
            # Generate resource ID
            resource_id = str(uuid.uuid4())
            
            # Create resource instance
            resource_class = self.resource_types[resource_type]
            resource = resource_class(
                resource_id=resource_id,
                name=name,
                project_id=project_id,
                **kwargs
            )
            
            # Validate resource
            resource.validate()
            
            # Convert to dictionary
            data = resource.to_dict()
            
            # Store in database
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO resources (
                        id, type, name, status, project_id,
                        created_at, updated_at, data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resource_id,
                        resource_type,
                        name,
                        resource.status,
                        project_id,
                        data["created_at"],
                        data["updated_at"],
                        json.dumps(data)
                    )
                )
                conn.commit()
                
            return data
            
        except Exception as e:
            raise ProcessingError(f"Error creating resource: {e}")
            
    def get_resource(
        self,
        resource_id: str
    ) -> Dict[str, Any]:
        """Get resource information.
        
        Args:
            resource_id: Resource ID.
            
        Returns:
            Resource information dictionary.
            
        Raises:
            ResourceNotFoundError: If resource not found.
            ProcessingError: If retrieval fails.
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data FROM resources WHERE id = ?",
                    (resource_id,)
                )
                result = cursor.fetchone()
                
            if not result:
                raise ResourceNotFoundError(f"Resource {resource_id} not found")
                
            return json.loads(result[0])
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error getting resource: {e}")
            
    def update_resource(
        self,
        resource_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update resource information.
        
        Args:
            resource_id: Resource ID.
            updates: Dictionary of updates.
            
        Returns:
            Updated resource information.
            
        Raises:
            ResourceNotFoundError: If resource not found.
            ValidationError: If updates are invalid.
            ProcessingError: If update fails.
        """
        try:
            # Get current resource
            current = self.get_resource(resource_id)
            
            # Create resource instance
            resource_class = self.resource_types[current["type"]]
            resource = resource_class.from_dict(current)
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(resource, key):
                    setattr(resource, key, value)
                    
            # Validate updated resource
            resource.validate()
            
            # Convert to dictionary
            data = resource.to_dict()
            data["updated_at"] = datetime.utcnow().isoformat()
            
            # Update database
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE resources
                    SET name = ?, status = ?, updated_at = ?, data = ?
                    WHERE id = ?
                    """,
                    (
                        data["name"],
                        data["status"],
                        data["updated_at"],
                        json.dumps(data),
                        resource_id
                    )
                )
                conn.commit()
                
            return data
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error updating resource: {e}")
            
    def delete_resource(self, resource_id: str) -> None:
        """Delete a resource.
        
        Args:
            resource_id: Resource ID.
            
        Raises:
            ResourceNotFoundError: If resource not found.
            ProcessingError: If deletion fails.
        """
        try:
            # Verify resource exists
            resource = self.get_resource(resource_id)
            
            # Delete from database
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM resources WHERE id = ?",
                    (resource_id,)
                )
                conn.commit()
                
            # Delete associated files
            resource_dir = (
                self.base_path / resource["type"] / resource_id
            )
            if resource_dir.exists():
                import shutil
                shutil.rmtree(resource_dir)
                
        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error deleting resource: {e}")
            
    def list_resources(
        self,
        resource_type: Optional[str] = None,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List resources with optional filtering.
        
        Args:
            resource_type: Optional resource type filter.
            project_id: Optional project ID filter.
            status: Optional status filter.
            filters: Optional additional filters.
            
        Returns:
            List of resource information dictionaries.
            
        Raises:
            ValidationError: If filter parameters are invalid.
        """
        try:
            # Build query
            query = "SELECT data FROM resources WHERE 1=1"
            params = []
            
            if resource_type:
                if resource_type not in self.resource_types:
                    raise ValidationError(f"Invalid resource type: {resource_type}")
                query += " AND type = ?"
                params.append(resource_type)
                
            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)
                
            if status:
                query += " AND status = ?"
                params.append(status)
                
            # Execute query
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                results = cursor.fetchall()
                
            # Parse results
            resources = [json.loads(row[0]) for row in results]
            
            # Apply additional filters
            if filters:
                for key, value in filters.items():
                    resources = [
                        r for r in resources
                        if r.get(key) == value
                    ]
                    
            return resources
            
        except Exception as e:
            raise ProcessingError(f"Error listing resources: {e}")
            
    def get_resource_file_path(
        self,
        resource_id: str,
        file_name: str
    ) -> Path:
        """Get path for resource file.
        
        Args:
            resource_id: Resource ID.
            file_name: File name.
            
        Returns:
            Path object for file.
            
        Raises:
            ResourceNotFoundError: If resource not found.
        """
        # Get resource
        resource = self.get_resource(resource_id)
        
        # Build path
        return (
            self.base_path / resource["type"] / 
            resource_id / file_name
        )
        
    def get_memory_usage(self) -> float:
        """Get manager memory usage.
        
        Returns:
            Memory usage in bytes.
        """
        total = 0
        
        # Database size
        db_path = self.base_path / "db" / "resources.db"
        if db_path.exists():
            total += db_path.stat().st_size
            
        # Resource directories
        for resource_type in self.resource_types:
            resource_dir = self.base_path / resource_type
            if resource_dir.exists():
                for entry in resource_dir.rglob("*"):
                    if entry.is_file():
                        total += entry.stat().st_size
                        
        return total 