"""Project Service for managing music generation projects."""

import os
import json
import shutil
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
from datetime import datetime
import uuid

from ..core.resources.base_resource import BaseResource
from ..core.resources.model_resource import ModelResource
from ..core.resources.track_resource import TrackResource
from ..core.resources.generation_resource import GenerationResource
from ..core.exceptions.base_exceptions import (
    ProcessingError,
    ValidationError,
    ResourceNotFoundError
)


class ProjectService:
    """Service for managing music generation projects.
    
    This class provides functionality for managing projects, including
    resource management, configuration handling, and output organization.
    """
    
    def __init__(
        self,
        base_path: Union[str, Path],
        resource_manager: Any,  # ResourceManager instance
        model_service: Any,  # ModelService instance
        generation_service: Any,  # GenerationService instance
        output_service: Any  # OutputService instance
    ):
        """Initialize the project service.
        
        Args:
            base_path: Base path for project storage.
            resource_manager: ResourceManager instance.
            model_service: ModelService instance.
            generation_service: GenerationService instance.
            output_service: OutputService instance.
        """
        self.base_path = Path(base_path)
        self.resource_manager = resource_manager
        self.model_service = model_service
        self.generation_service = generation_service
        self.output_service = output_service
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Ensure project directories exist
        self._ensure_directories()
        
    def _ensure_directories(self) -> None:
        """Ensure required project directories exist."""
        directories = [
            self.base_path,
            self.base_path / "projects",
            self.base_path / "shared_models",
            self.base_path / "shared_resources"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new project.
        
        Args:
            name: Project name.
            description: Optional project description.
            metadata: Optional project metadata.
            
        Returns:
            Project information dictionary.
            
        Raises:
            ValidationError: If project parameters are invalid.
            ProcessingError: If project creation fails.
        """
        try:
            # Validate name
            if not name or not name.strip():
                raise ValidationError("Project name cannot be empty")
                
            # Generate project ID and path
            project_id = str(uuid.uuid4())
            project_path = self.base_path / "projects" / project_id
            
            # Create project structure
            project_dirs = [
                project_path,
                project_path / "models",
                project_path / "tracks",
                project_path / "generations",
                project_path / "outputs",
                project_path / "config"
            ]
            
            for directory in project_dirs:
                directory.mkdir(parents=True, exist_ok=True)
                
            # Create project config
            config = {
                "id": project_id,
                "name": name,
                "description": description or "",
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "status": "active"
            }
            
            # Save config
            config_path = project_path / "config" / "project.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                
            return config
            
        except Exception as e:
            raise ProcessingError(f"Error creating project: {e}")
            
    def get_project(self, project_id: str) -> Dict[str, Any]:
        """Get project information.
        
        Args:
            project_id: Project ID.
            
        Returns:
            Project information dictionary.
            
        Raises:
            ResourceNotFoundError: If project not found.
            ProcessingError: If loading project fails.
        """
        try:
            project_path = self.base_path / "projects" / project_id
            config_path = project_path / "config" / "project.json"
            
            if not config_path.exists():
                raise ResourceNotFoundError(f"Project {project_id} not found")
                
            with open(config_path, "r") as f:
                config = json.load(f)
                
            # Add additional info
            config.update({
                "path": str(project_path),
                "size": self._get_directory_size(project_path),
                "resource_counts": self._get_resource_counts(project_id)
            })
            
            return config
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error loading project: {e}")
            
    def list_projects(
        self,
        status: Optional[str] = None,
        sort_by: str = "updated_at",
        reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """List available projects.
        
        Args:
            status: Optional status filter.
            sort_by: Field to sort by.
            reverse: Whether to reverse sort order.
            
        Returns:
            List of project information dictionaries.
        """
        projects = []
        projects_dir = self.base_path / "projects"
        
        for project_dir in projects_dir.iterdir():
            if project_dir.is_dir():
                try:
                    project = self.get_project(project_dir.name)
                    if not status or project["status"] == status:
                        projects.append(project)
                except Exception as e:
                    self.logger.warning(f"Error loading project {project_dir.name}: {e}")
                    
        # Sort projects
        if sort_by in projects[0] if projects else False:
            projects.sort(key=lambda x: x[sort_by], reverse=reverse)
            
        return projects
        
    def update_project(
        self,
        project_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update project information.
        
        Args:
            project_id: Project ID.
            updates: Dictionary of updates.
            
        Returns:
            Updated project information.
            
        Raises:
            ResourceNotFoundError: If project not found.
            ValidationError: If updates are invalid.
            ProcessingError: If update fails.
        """
        try:
            # Get current config
            project = self.get_project(project_id)
            
            # Validate updates
            invalid_keys = set(updates.keys()) - {
                "name", "description", "metadata", "status"
            }
            if invalid_keys:
                raise ValidationError(f"Invalid update keys: {invalid_keys}")
                
            # Update config
            project.update(updates)
            project["updated_at"] = datetime.utcnow().isoformat()
            
            # Save config
            config_path = (
                self.base_path / "projects" / project_id / 
                "config" / "project.json"
            )
            with open(config_path, "w") as f:
                # Only save persistent fields
                config = {
                    k: v for k, v in project.items()
                    if k not in ["path", "size", "resource_counts"]
                }
                json.dump(config, f, indent=2)
                
            return project
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error updating project: {e}")
            
    def delete_project(
        self,
        project_id: str,
        delete_resources: bool = False
    ) -> None:
        """Delete a project.
        
        Args:
            project_id: Project ID.
            delete_resources: Whether to delete project resources.
            
        Raises:
            ResourceNotFoundError: If project not found.
            ProcessingError: If deletion fails.
        """
        try:
            project_path = self.base_path / "projects" / project_id
            
            if not project_path.exists():
                raise ResourceNotFoundError(f"Project {project_id} not found")
                
            if delete_resources:
                # Delete all project resources
                resource_types = ["models", "tracks", "generations"]
                for resource_type in resource_types:
                    resources = self.list_project_resources(
                        project_id,
                        resource_type
                    )
                    for resource in resources:
                        self.resource_manager.delete_resource(resource["id"])
                        
            # Delete project directory
            shutil.rmtree(project_path)
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error deleting project: {e}")
            
    def list_project_resources(
        self,
        project_id: str,
        resource_type: str,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List project resources.
        
        Args:
            project_id: Project ID.
            resource_type: Type of resource ("models", "tracks", "generations").
            status: Optional status filter.
            
        Returns:
            List of resource information dictionaries.
            
        Raises:
            ValidationError: If resource type is invalid.
            ResourceNotFoundError: If project not found.
        """
        if resource_type not in ["models", "tracks", "generations"]:
            raise ValidationError(f"Invalid resource type: {resource_type}")
            
        # Verify project exists
        if not (self.base_path / "projects" / project_id).exists():
            raise ResourceNotFoundError(f"Project {project_id} not found")
            
        # Get resources from resource manager
        resources = self.resource_manager.list_resources(
            resource_type=resource_type,
            filters={"project_id": project_id}
        )
        
        # Filter by status if specified
        if status:
            resources = [r for r in resources if r["status"] == status]
            
        return resources
        
    def get_project_resource(
        self,
        project_id: str,
        resource_id: str
    ) -> Dict[str, Any]:
        """Get project resource information.
        
        Args:
            project_id: Project ID.
            resource_id: Resource ID.
            
        Returns:
            Resource information dictionary.
            
        Raises:
            ResourceNotFoundError: If project or resource not found.
        """
        # Verify project exists
        if not (self.base_path / "projects" / project_id).exists():
            raise ResourceNotFoundError(f"Project {project_id} not found")
            
        # Get resource from resource manager
        resource = self.resource_manager.get_resource(resource_id)
        
        # Verify resource belongs to project
        if resource["project_id"] != project_id:
            raise ResourceNotFoundError(
                f"Resource {resource_id} not found in project {project_id}"
            )
            
        return resource
        
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total
        
    def _get_resource_counts(self, project_id: str) -> Dict[str, int]:
        """Get counts of different resource types in project."""
        counts = {}
        resource_types = ["models", "tracks", "generations"]
        
        for resource_type in resource_types:
            try:
                resources = self.list_project_resources(
                    project_id,
                    resource_type
                )
                counts[resource_type] = len(resources)
            except Exception:
                counts[resource_type] = 0
                
        return counts
        
    def get_memory_usage(self) -> float:
        """Get service memory usage.
        
        Returns:
            Memory usage in bytes.
        """
        # Get size of all project directories
        total = 0
        projects_dir = self.base_path / "projects"
        
        if projects_dir.exists():
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir():
                    total += self._get_directory_size(project_dir)
                    
        return total 