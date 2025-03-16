"""Service for generating musical content."""
import logging
from typing import Dict, Any, List, Optional

from ...core.base.base_model import BaseAIModel
from ...core.config.model_config import GenerationConfig
from ...core.exceptions.base_exceptions import ModelNotFoundError, ProcessingError
from ...core.resources.base_resources import ProjectResource, TrackResource
from ...utils.metrics.prometheus_metrics import MODEL_INFERENCE_TIME, GENERATION_SIZES


class GenerationService:
    """Service for generating musical content."""
    
    def __init__(
        self, 
        model_service: 'ModelService',
        resource_service: 'ResourceService',
        output_service: 'OutputService'
    ):
        """Initialize the generation service.
        
        Args:
            model_service: Service for accessing AI models.
            resource_service: Service for accessing resources.
            output_service: Service for managing outputs.
        """
        self.model_service = model_service
        self.resource_service = resource_service
        self.output_service = output_service
        self.logger = logging.getLogger(__name__)
        
    async def generate_from_text(
        self,
        text: str,
        config: GenerationConfig,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate music from text description.
        
        Args:
            text: Text description of desired music.
            config: Generation configuration.
            project_id: Optional project ID to associate with generation.
            
        Returns:
            Dictionary containing generation results.
        """
        self.logger.info(f"Generating music from text: {text[:50]}...")
        
        # Get appropriate model
        model = await self.model_service.get_model(
            model_type="text_to_music",
            config=config
        )
        
        # Prepare inputs
        inputs = {"text": text, "config": config.dict()}
        
        # Generate outputs with timing
        with MODEL_INFERENCE_TIME.labels(
            model_type="text_to_music",
            model_id=model.config.model_id
        ).time():
            outputs = await model.predict(inputs)
        
        # Track generation sizes
        for content_type, content in outputs.items():
            if isinstance(content, (bytes, bytearray)):
                GENERATION_SIZES.labels(content_type=content_type).observe(len(content))
        
        # Save results
        result_id = await self.output_service.save_generation(
            outputs=outputs,
            generation_type="text_to_music",
            source_text=text,
            project_id=project_id,
            config=config
        )
        
        return {
            "result_id": result_id,
            "outputs": outputs
        }
        
    async def generate_from_midi(
        self,
        midi_data: bytes,
        config: GenerationConfig,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate variations from MIDI input.
        
        Args:
            midi_data: Input MIDI data.
            config: Generation configuration.
            project_id: Optional project ID to associate with generation.
            
        Returns:
            Dictionary containing generation results.
        """
        self.logger.info("Generating variations from MIDI input...")
        
        # Get appropriate model
        model = await self.model_service.get_model(
            model_type="midi_variation",
            config=config
        )
        
        # Prepare inputs
        inputs = {"midi": midi_data, "config": config.dict()}
        
        # Generate outputs with timing
        with MODEL_INFERENCE_TIME.labels(
            model_type="midi_variation",
            model_id=model.config.model_id
        ).time():
            outputs = await model.predict(inputs)
        
        # Track generation sizes
        for content_type, content in outputs.items():
            if isinstance(content, (bytes, bytearray)):
                GENERATION_SIZES.labels(content_type=content_type).observe(len(content))
        
        # Save results
        result_id = await self.output_service.save_generation(
            outputs=outputs,
            generation_type="midi_variation",
            source_midi=midi_data,
            project_id=project_id,
            config=config
        )
        
        return {
            "result_id": result_id,
            "outputs": outputs
        }
        
    async def generate_from_audio(
        self,
        audio_data: bytes,
        config: GenerationConfig,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate variations from audio input.
        
        Args:
            audio_data: Input audio data.
            config: Generation configuration.
            project_id: Optional project ID to associate with generation.
            
        Returns:
            Dictionary containing generation results.
        """
        self.logger.info("Generating variations from audio input...")
        
        # Get appropriate model
        model = await self.model_service.get_model(
            model_type="audio_variation",
            config=config
        )
        
        # Prepare inputs
        inputs = {"audio": audio_data, "config": config.dict()}
        
        # Generate outputs with timing
        with MODEL_INFERENCE_TIME.labels(
            model_type="audio_variation",
            model_id=model.config.model_id
        ).time():
            outputs = await model.predict(inputs)
        
        # Track generation sizes
        for content_type, content in outputs.items():
            if isinstance(content, (bytes, bytearray)):
                GENERATION_SIZES.labels(content_type=content_type).observe(len(content))
        
        # Save results
        result_id = await self.output_service.save_generation(
            outputs=outputs,
            generation_type="audio_variation",
            source_audio=audio_data,
            project_id=project_id,
            config=config
        )
        
        return {
            "result_id": result_id,
            "outputs": outputs
        } 