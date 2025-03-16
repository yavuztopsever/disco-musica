"""FastAPI application for Disco-Musica."""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.openapi.docs import get_swagger_ui_html
from typing import Dict, Any, List, Optional

from ...services.generation_service import GenerationService
from ...services.model_service import ModelService
from ...services.resource_service import ResourceService
from ...core.config.model_config import GenerationConfig
from ...core.exceptions.base_exceptions import DiscoMusicaError, ResourceNotFoundError


def create_api(
    generation_service: GenerationService,
    model_service: ModelService,
    resource_service: ResourceService
) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        generation_service: Service for music generation.
        model_service: Service for model management.
        resource_service: Service for resource management.
        
    Returns:
        FastAPI application.
    """
    app = FastAPI(
        title="Disco-Musica API",
        description="API for the Disco-Musica music generation system",
        version="1.0.0",
        docs_url=None  # Disable default docs
    )
    
    @app.get("/docs", include_in_schema=False)
    async def custom_docs():
        """Custom SwaggerUI with branding."""
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="Disco-Musica API Documentation",
            swagger_ui_parameters={
                "syntaxHighlight.theme": "monokai",
                "docExpansion": "none",
                "defaultModelsExpandDepth": -1,
                "tryItOutEnabled": True,
            }
        )
    
    @app.post("/api/generate/text-to-music")
    async def generate_from_text(
        text: str,
        config: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """Generate music from text description.
        
        Args:
            text: Text description.
            config: Optional generation configuration.
            
        Returns:
            Generation results.
        """
        if config is None:
            config = GenerationConfig()
            
        try:
            result = await generation_service.generate_from_text(
                text=text,
                config=config
            )
            return result
        except DiscoMusicaError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    @app.get("/api/resources/{resource_id}")
    async def get_resource(resource_id: str) -> Dict[str, Any]:
        """Get a resource by ID.
        
        Args:
            resource_id: Resource ID.
            
        Returns:
            Resource data.
        """
        try:
            resource = await resource_service.get_resource(resource_id)
            return resource.dict()
        except ResourceNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Resource {resource_id} not found"
            )
            
    @app.get("/api/models")
    async def list_models() -> List[Dict[str, Any]]:
        """List available models.
        
        Returns:
            List of model information.
        """
        models = await model_service.list_models()
        return [model.dict() for model in models]
        
    return app 