"""Integration service for handling external tool integrations."""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ...core.exceptions import DiscoMusicaError, ResourceNotFoundError
from ...core.resources import ProjectResource, TrackResource
from ..generation_service import GenerationService
from ..model_service import ModelService
from ..resource_service import ResourceService
from ..output_service import OutputService

logger = logging.getLogger(__name__)

class LogicProMessage(BaseModel):
    """Message format for Logic Pro communication."""
    command: str
    params: Dict[str, Any]
    timestamp: datetime = datetime.utcnow()

class IntegrationService:
    """Service for managing external tool integrations."""
    
    def __init__(
        self,
        generation_service: GenerationService,
        model_service: ModelService,
        resource_service: ResourceService,
        output_service: OutputService
    ):
        """Initialize the integration service.
        
        Args:
            generation_service: Service for music generation.
            model_service: Service for model management.
            resource_service: Service for resource management.
            output_service: Service for output management.
        """
        self.generation_service = generation_service
        self.model_service = model_service
        self.resource_service = resource_service
        self.output_service = output_service
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger(__name__)
        
    async def register_endpoints(self, app: FastAPI):
        """Register WebSocket endpoints with FastAPI app.
        
        Args:
            app: FastAPI application.
        """
        @app.websocket("/ws/logic-pro")
        async def logic_pro_websocket(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    data = await websocket.receive_json()
                    message = LogicProMessage(**data)
                    
                    # Process command
                    response = await self.process_command(message.command, message.params)
                    
                    # Send response
                    await websocket.send_json(response)
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                error_response = {
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                await websocket.send_json(error_response)
                
    async def process_command(
        self,
        command: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a command from Logic Pro.
        
        Args:
            command: Command to process.
            params: Command parameters.
            
        Returns:
            Response dictionary.
        """
        handlers = {
            "project_info": self.handle_project_info,
            "quantize_project": self.handle_quantize_project,
            "generate_tracks": self.handle_generate_tracks,
            "tune_vocals": self.handle_tune_vocals,
            "apply_effects": self.handle_apply_effects,
            "apply_mastering": self.handle_apply_mastering
        }
        
        handler = handlers.get(command)
        if not handler:
            return {
                "status": "error",
                "error": f"Unknown command: {command}"
            }
            
        try:
            result = await handler(params)
            return {
                "status": "success",
                "result": result
            }
        except DiscoMusicaError as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
            
    async def handle_project_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle project info command.
        
        Args:
            params: Command parameters.
            
        Returns:
            Project information.
        """
        project_path = params.get("project_path")
        
        # Create or update project profile
        project_id = await self.resource_service.create_or_update_project(
            project_path=project_path
        )
        
        # Get project information
        project = await self.resource_service.get_project(project_id)
        
        return {
            "project_id": project_id,
            "project_info": project.dict()
        }
        
    async def handle_quantize_project(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle project quantization command.
        
        Args:
            params: Command parameters.
            
        Returns:
            Quantization results.
        """
        project_id = params.get("project_id")
        quantization_params = params.get("quantization_params", {})
        
        # Get project
        project = await self.resource_service.get_project(project_id)
        if not project:
            raise ResourceNotFoundError(f"Project {project_id} not found")
            
        # Apply quantization
        result = await self.generation_service.quantize_project(
            project_id=project_id,
            **quantization_params
        )
        
        return {
            "project_id": project_id,
            "quantization_result": result
        }
        
    async def handle_generate_tracks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle track generation command.
        
        Args:
            params: Command parameters.
            
        Returns:
            Generation results.
        """
        project_id = params.get("project_id")
        generation_params = params.get("generation_params", {})
        
        # Get project
        project = await self.resource_service.get_project(project_id)
        if not project:
            raise ResourceNotFoundError(f"Project {project_id} not found")
            
        # Generate tracks
        result = await self.generation_service.generate_tracks(
            project_id=project_id,
            **generation_params
        )
        
        return {
            "project_id": project_id,
            "generation_result": result
        }
        
    async def handle_tune_vocals(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle vocal tuning command.
        
        Args:
            params: Command parameters.
            
        Returns:
            Tuning results.
        """
        project_id = params.get("project_id")
        track_id = params.get("track_id")
        tuning_params = params.get("tuning_params", {})
        
        # Get project and track
        project = await self.resource_service.get_project(project_id)
        if not project:
            raise ResourceNotFoundError(f"Project {project_id} not found")
            
        track = await self.resource_service.get_track(track_id)
        if not track:
            raise ResourceNotFoundError(f"Track {track_id} not found")
            
        # Apply vocal tuning
        result = await self.generation_service.tune_vocals(
            project_id=project_id,
            track_id=track_id,
            **tuning_params
        )
        
        return {
            "project_id": project_id,
            "track_id": track_id,
            "tuning_result": result
        }
        
    async def handle_apply_effects(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle effect application command.
        
        Args:
            params: Command parameters.
            
        Returns:
            Effect application results.
        """
        project_id = params.get("project_id")
        track_id = params.get("track_id")
        effect_params = params.get("effect_params", {})
        
        # Get project and track
        project = await self.resource_service.get_project(project_id)
        if not project:
            raise ResourceNotFoundError(f"Project {project_id} not found")
            
        track = await self.resource_service.get_track(track_id)
        if not track:
            raise ResourceNotFoundError(f"Track {track_id} not found")
            
        # Apply effects
        result = await self.generation_service.apply_effects(
            project_id=project_id,
            track_id=track_id,
            **effect_params
        )
        
        return {
            "project_id": project_id,
            "track_id": track_id,
            "effect_result": result
        }
        
    async def handle_apply_mastering(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mastering command.
        
        Args:
            params: Command parameters.
            
        Returns:
            Mastering results.
        """
        project_id = params.get("project_id")
        mastering_params = params.get("mastering_params", {})
        
        # Get project
        project = await self.resource_service.get_project(project_id)
        if not project:
            raise ResourceNotFoundError(f"Project {project_id} not found")
            
        # Apply mastering
        result = await self.generation_service.apply_mastering(
            project_id=project_id,
            **mastering_params
        )
        
        return {
            "project_id": project_id,
            "mastering_result": result
        }
        
    async def broadcast_update(self, message: Dict[str, Any]):
        """Broadcast update to all connected clients.
        
        Args:
            message: Message to broadcast.
        """
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {e}")
                self.active_connections.remove(connection) 