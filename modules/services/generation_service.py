"""
Generation Service module for Disco Musica.

This module provides services for music generation across different modalities.
"""

import os
import time
import datetime
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json

import numpy as np
import asyncio
import logging

from modules.core.config import config
from modules.core.audio_processor import AudioProcessor
from modules.services.model_service import get_model_service
from modules.core.text_to_music_model import TextToMusicModel
from modules.core.processors.processor_manager import ProcessorManager
from modules.core.resources.resource_manager import ResourceManager
from modules.core.resources.generation_resource import GenerationResource
from modules.core.resources.model_resource import ModelResource
from modules.core.resources.track_resource import TrackResource
from modules.models.model_registry import ModelRegistry
from modules.exceptions.base_exceptions import (
    ProcessingError,
    ValidationError,
    ResourceNotFoundError
)
from ..core.processors.audio_processor import AudioProcessor
from ..core.processors.midi_processor import MIDIProcessor
from ..core.processors.text_processor import TextProcessor
from ..services.project_service import ProjectService
from ..services.output_service import OutputService
from ..services.model_service import ModelService


class GenerationService:
    """
    Service for music generation across different modalities.
    
    This class provides a unified interface for generating music from text,
    audio, MIDI, and image inputs, handling the model selection, generation,
    and output processing.
    """
    
    def __init__(
        self,
        base_path: Path,
        project_service: ProjectService,
        output_service: OutputService,
        model_service: ModelService
    ):
        """
        Initialize the GenerationService.
        
        Args:
            base_path: Base path for file operations.
            project_service: Project service instance.
            output_service: Output service instance.
            model_service: Model service instance.
        """
        self.base_path = base_path
        self.project_service = project_service
        self.output_service = output_service
        self.model_service = model_service
        
        # Initialize processors
        self.audio_processor = AudioProcessor()
        self.midi_processor = MIDIProcessor()
        self.text_processor = TextProcessor()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Output directory
        self.output_dir = Path(config.get("paths", "output_dir", "outputs"))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generation history
        self.history = {}
    
    async def generate_from_text(
        self,
        prompt: str,
        track_id: str,
        model_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> GenerationResource:
        """
        Generate music from text prompt.
        
        Args:
            prompt: Text prompt.
            track_id: Parent track ID.
            model_id: Model ID to use.
            config: Optional generation configuration.
            
        Returns:
            Generation resource.
            
        Raises:
            ResourceNotFoundError: If track or model not found.
            ProcessingError: If generation fails.
        """
        try:
            # Get track and model
            track = self.resource_manager.get_resource("track", track_id)
            model = self.resource_manager.get_resource("model", model_id)
            
            # Process prompt
            processed_prompt = self.processor_manager.process_generation_prompt(
                prompt
            )
            
            # Create generation resource
            generation = self.resource_manager.create_resource(
                "generation",
                generation_type="text_to_music",
                track_id=track_id
            )
            
            # Set generation metadata
            generation.set_model(model_id)
            generation.set_config(config or {})
            generation.set_inputs({"prompt": processed_prompt})
            
            # Get model instance
            model_instance = self.model_registry.get_model(
                model.model_type,
                model.version
            )
            
            # Generate audio
            try:
                output = await model_instance.predict({
                    "text": processed_prompt,
                    **config or {}
                })
                
                # Save output file
                output_path = self.output_dir / f"{generation.resource_id}.wav"
                self.processor_manager.audio.save_audio(
                    output["audio"],
                    output_path
                )
                
                # Update generation
                file_info = self.processor_manager.audio.get_file_info(output_path)
                generation.set_file(
                    str(output_path),
                    file_info["duration"]
                )
                generation.set_outputs({
                    "file_path": str(output_path),
                    "duration": file_info["duration"]
                })
                
                # Update track
                track.add_generation(generation.resource_id)
                
                return generation
                
            except Exception as e:
                generation.delete()
                raise ProcessingError(f"Generation failed: {e}")
                
        except Exception as e:
            raise ProcessingError(f"Error in text generation: {e}")
    
    def generate_from_audio(
        self,
        audio_path: Union[str, Path],
        prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        duration: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        save_output: bool = True,
        output_format: str = "wav"
    ) -> Dict:
        """
        Generate music from an audio input.
        
        Args:
            audio_path: Path to the input audio file.
            prompt: Optional text prompt to guide generation.
            model_id: ID of the model to use. If None, the default model is used.
            duration: Duration of the generated audio in seconds.
            temperature: Sampling temperature (higher = more random).
            seed: Random seed for reproducibility.
            save_output: Whether to save the generated audio.
            output_format: Format to save the audio in.
            
        Returns:
            Dictionary containing generation results.
        """
        # Start timing
        start_time = time.time()
        
        # Get or create generation ID
        generation_id = str(uuid.uuid4())
        
        # Get default model if not specified
        if model_id is None:
            model_id = self.model_service.get_default_model_for_task("audio_to_music")
            if model_id is None:
                raise ValueError("No default model found for audio-to-music generation")
        
        # Set seed if specified
        if seed is not None:
            import torch
            import random
            import numpy as np
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        
        try:
            # Create model instance
            model = self.model_service.create_model_instance(model_id)
            if model is None:
                raise ValueError(f"Failed to create model instance for {model_id}")
            
            # Generate audio
            audio_data = model.generate(
                audio_input=audio_path,
                prompt=prompt,
                duration=duration,
                temperature=temperature
            )
            
            # Get model parameters
            sample_rate = model.metadata["parameters"].get("sample_rate", 44100)
            
            # Save output if requested
            output_path = None
            if save_output:
                # Create output directory
                output_subdir = self.output_dir / "audio_to_music"
                os.makedirs(output_subdir, exist_ok=True)
                
                # Generate filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"audio_to_music_{timestamp}.{output_format}"
                output_path = output_subdir / filename
                
                # Save audio
                self.audio_processor.save_audio(
                    audio_data=audio_data,
                    output_path=output_path,
                    sr=sample_rate,
                    format=output_format
                )
            
            # Analyze audio
            audio_analysis = self.audio_processor.analyze_audio(audio_data, sample_rate)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Prepare result
            result = {
                "generation_id": generation_id,
                "model_id": model_id,
                "prompt": prompt,
                "input_path": str(audio_path),
                "audio_data": audio_data,
                "sample_rate": sample_rate,
                "duration": len(audio_data) / sample_rate,
                "output_path": str(output_path) if output_path else None,
                "generation_time": generation_time,
                "parameters": {
                    "duration": duration,
                    "temperature": temperature,
                    "seed": seed
                },
                "analysis": audio_analysis
            }
            
            # Save to history
            self.history[generation_id] = {
                "type": "audio_to_music",
                "timestamp": datetime.datetime.now().isoformat(),
                "model_id": model_id,
                "prompt": prompt,
                "input_path": str(audio_path),
                "output_path": str(output_path) if output_path else None,
                "parameters": result["parameters"]
            }
            
            return result
        
        except Exception as e:
            print(f"Error in audio-to-music generation: {e}")
            raise
    
    def generate_from_midi(
        self,
        midi_path: Union[str, Path],
        prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        instrument_prompt: Optional[str] = None,
        duration: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        save_output: bool = True,
        output_format: str = "wav"
    ) -> Dict:
        """
        Generate music from a MIDI file.
        
        Args:
            midi_path: Path to the input MIDI file.
            prompt: Optional text prompt to guide generation.
            model_id: ID of the model to use. If None, the default model is used.
            instrument_prompt: Optional prompt describing the instrumentation.
            duration: Duration of the generated audio in seconds.
            temperature: Sampling temperature (higher = more random).
            seed: Random seed for reproducibility.
            save_output: Whether to save the generated audio.
            output_format: Format to save the audio in.
            
        Returns:
            Dictionary containing generation results.
        """
        # Start timing
        start_time = time.time()
        
        # Get or create generation ID
        generation_id = str(uuid.uuid4())
        
        # Get default model if not specified
        if model_id is None:
            model_id = self.model_service.get_default_model_for_task("midi_to_audio")
            if model_id is None:
                raise ValueError("No default model found for MIDI-to-audio generation")
        
        # Set seed if specified
        if seed is not None:
            import torch
            import random
            import numpy as np
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        
        try:
            # Create model instance
            model = self.model_service.create_model_instance(model_id)
            if model is None:
                raise ValueError(f"Failed to create model instance for {model_id}")
            
            # Generate audio
            audio_data = model.generate(
                midi_input=midi_path,
                prompt=prompt,
                instrument_prompt=instrument_prompt,
                duration=duration,
                temperature=temperature
            )
            
            # Get model parameters
            sample_rate = model.metadata["parameters"].get("sample_rate", 44100)
            
            # Save output if requested
            output_path = None
            if save_output:
                # Create output directory
                output_subdir = self.output_dir / "midi_to_audio"
                os.makedirs(output_subdir, exist_ok=True)
                
                # Generate filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"midi_to_audio_{timestamp}.{output_format}"
                output_path = output_subdir / filename
                
                # Save audio
                self.audio_processor.save_audio(
                    audio_data=audio_data,
                    output_path=output_path,
                    sr=sample_rate,
                    format=output_format
                )
            
            # Analyze audio
            audio_analysis = self.audio_processor.analyze_audio(audio_data, sample_rate)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Prepare result
            result = {
                "generation_id": generation_id,
                "model_id": model_id,
                "prompt": prompt,
                "instrument_prompt": instrument_prompt,
                "input_path": str(midi_path),
                "audio_data": audio_data,
                "sample_rate": sample_rate,
                "duration": len(audio_data) / sample_rate,
                "output_path": str(output_path) if output_path else None,
                "generation_time": generation_time,
                "parameters": {
                    "duration": duration,
                    "temperature": temperature,
                    "seed": seed
                },
                "analysis": audio_analysis
            }
            
            # Save to history
            self.history[generation_id] = {
                "type": "midi_to_audio",
                "timestamp": datetime.datetime.now().isoformat(),
                "model_id": model_id,
                "prompt": prompt,
                "instrument_prompt": instrument_prompt,
                "input_path": str(midi_path),
                "output_path": str(output_path) if output_path else None,
                "parameters": result["parameters"]
            }
            
            return result
        
        except Exception as e:
            print(f"Error in MIDI-to-audio generation: {e}")
            raise
    
    def generate_from_image(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        duration: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        save_output: bool = True,
        output_format: str = "wav"
    ) -> Dict:
        """
        Generate music from an image.
        
        Args:
            image_path: Path to the input image file.
            prompt: Optional text prompt to guide generation.
            model_id: ID of the model to use. If None, the default model is used.
            duration: Duration of the generated audio in seconds.
            temperature: Sampling temperature (higher = more random).
            seed: Random seed for reproducibility.
            save_output: Whether to save the generated audio.
            output_format: Format to save the audio in.
            
        Returns:
            Dictionary containing generation results.
        """
        # This is a placeholder for image-to-music generation.
        # In a real implementation, this would call an ImageToMusicModel.
        print("Image-to-music generation is not yet implemented")
        
        # Start timing
        start_time = time.time()
        
        # Get or create generation ID
        generation_id = str(uuid.uuid4())
        
        # Get default model if not specified
        if model_id is None:
            model_id = self.model_service.get_default_model_for_task("text_to_music")
            if model_id is None:
                raise ValueError("No default model found for text-to-music generation")
        
        try:
            # For now, use image filename to create a prompt
            image_path = Path(image_path)
            if prompt is None:
                image_name = image_path.stem.replace("_", " ").replace("-", " ")
                prompt = f"Generate music inspired by an image of {image_name}"
            
            # Use text-to-music generation as a fallback
            return self.generate_from_text(
                prompt=prompt,
                model_id=model_id,
                duration=duration,
                temperature=temperature,
                seed=seed,
                save_output=save_output,
                output_format=output_format
            )
        
        except Exception as e:
            print(f"Error in image-to-music generation: {e}")
            raise
    
    def get_generation_history(self) -> Dict[str, Dict]:
        """
        Get the generation history.
        
        Returns:
            Dictionary of generation history entries.
        """
        return self.history
    
    def get_generation_by_id(self, generation_id: str) -> Optional[Dict]:
        """
        Get a specific generation entry by ID.
        
        Args:
            generation_id: ID of the generation.
            
        Returns:
            Generation entry or None if not found.
        """
        return self.history.get(generation_id)
    
    def clear_history(self) -> None:
        """
        Clear the generation history.
        """
        self.history = {}
    
    def generate_variations(
        self,
        generation_id: str,
        num_variations: int = 3,
        variation_params: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Generate variations of a previous generation.
        
        Args:
            generation_id: ID of the previous generation.
            num_variations: Number of variations to generate.
            variation_params: Parameters to override for variations.
            
        Returns:
            List of variation results.
        """
        # Get the original generation
        original = self.get_generation_by_id(generation_id)
        if not original:
            raise ValueError(f"Generation {generation_id} not found")
        
        variations = []
        
        if variation_params is None:
            variation_params = {}
        
        # Create variations with different parameters
        for i in range(num_variations):
            # Vary the temperature for each variation
            params = original["parameters"].copy()
            params.update(variation_params)
            
            # Add some randomness to temperature if not explicitly set
            if "temperature" not in variation_params:
                base_temp = params.get("temperature", 1.0)
                params["temperature"] = base_temp * (0.8 + 0.4 * np.random.random())
            
            # Use a different seed for each variation
            params["seed"] = np.random.randint(0, 2**32 - 1)
            
            # Generate variation
            if original["type"] == "text_to_music":
                result = self.generate_from_text(
                    prompt=original["prompt"],
                    model_id=original["model_id"],
                    **params
                )
            elif original["type"] == "audio_to_music":
                result = self.generate_from_audio(
                    audio_path=original["input_path"],
                    prompt=original.get("prompt"),
                    model_id=original["model_id"],
                    **params
                )
            elif original["type"] == "midi_to_audio":
                result = self.generate_from_midi(
                    midi_path=original["input_path"],
                    prompt=original.get("prompt"),
                    model_id=original["model_id"],
                    **params
                )
            elif original["type"] == "image_to_music":
                result = self.generate_from_image(
                    image_path=original["input_path"],
                    prompt=original.get("prompt"),
                    model_id=original["model_id"],
                    **params
                )
            
            # Add variation metadata
            result["variation_of"] = generation_id
            result["variation_number"] = i + 1
            
            variations.append(result)
        
        return variations

    async def generate_variation(
        self,
        track_id: str,
        model_id: str,
        variation_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> GenerationResource:
        """Generate variation of existing track.
        
        Args:
            track_id: Parent track ID.
            model_id: Model ID to use.
            variation_type: Type of variation ("midi" or "audio").
            config: Optional generation configuration.
            
        Returns:
            Generation resource.
            
        Raises:
            ResourceNotFoundError: If track or model not found.
            ValidationError: If variation type is invalid.
            ProcessingError: If generation fails.
        """
        if variation_type not in ["midi", "audio"]:
            raise ValidationError(f"Invalid variation type: {variation_type}")
            
        try:
            # Get track and model
            track = self.resource_manager.get_resource("track", track_id)
            model = self.resource_manager.get_resource("model", model_id)
            
            if not track.file_path:
                raise ValidationError("Track has no source file")
                
            # Create generation resource
            generation = self.resource_manager.create_resource(
                "generation",
                generation_type=f"{variation_type}_variation",
                track_id=track_id
            )
            
            # Set generation metadata
            generation.set_model(model_id)
            generation.set_config(config or {})
            generation.set_inputs({
                "source_file": track.file_path,
                "source_format": track.file_format
            })
            
            # Get model instance
            model_instance = self.model_registry.get_model(
                model.model_type,
                model.version
            )
            
            # Generate variation
            try:
                # Load source file
                if variation_type == "midi":
                    source = self.processor_manager.midi.load_midi(track.file_path)
                else:  # audio
                    source = self.processor_manager.audio.load_audio(track.file_path)
                    
                # Generate variation
                output = await model_instance.predict({
                    "variation_type": source,
                    **config or {}
                })
                
                # Save output file
                output_path = self.output_dir / f"{generation.resource_id}.{track.file_format}"
                if variation_type == "midi":
                    self.processor_manager.midi.save_midi(
                        output["midi"],
                        output_path
                    )
                else:  # audio
                    self.processor_manager.audio.save_audio(
                        output["audio"],
                        output_path
                    )
                    
                # Update generation
                file_info = self.processor_manager.analyze_file(
                    output_path,
                    "duration"
                )
                generation.set_file(
                    str(output_path),
                    file_info["duration"]
                )
                generation.set_outputs({
                    "file_path": str(output_path),
                    "duration": file_info["duration"]
                })
                
                # Update track
                track.add_generation(generation.resource_id)
                
                return generation
                
            except Exception as e:
                generation.delete()
                raise ProcessingError(f"Generation failed: {e}")
                
        except Exception as e:
            raise ProcessingError(f"Error in variation generation: {e}")
            
    async def combine_generations(
        self,
        generation_ids: List[str],
        track_id: str,
        model_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> GenerationResource:
        """Combine multiple generations.
        
        Args:
            generation_ids: List of generation IDs to combine.
            track_id: Parent track ID.
            model_id: Model ID to use.
            config: Optional combination configuration.
            
        Returns:
            Generation resource.
            
        Raises:
            ResourceNotFoundError: If resources not found.
            ValidationError: If generations are incompatible.
            ProcessingError: If combination fails.
        """
        try:
            # Get resources
            track = self.resource_manager.get_resource("track", track_id)
            model = self.resource_manager.get_resource("model", model_id)
            generations = [
                self.resource_manager.get_resource("generation", gen_id)
                for gen_id in generation_ids
            ]
            
            # Validate generations
            if not all(gen.file_path for gen in generations):
                raise ValidationError("All generations must have output files")
                
            # Create generation resource
            generation = self.resource_manager.create_resource(
                "generation",
                generation_type="combination",
                track_id=track_id
            )
            
            # Set generation metadata
            generation.set_model(model_id)
            generation.set_config(config or {})
            generation.set_inputs({
                "source_generations": generation_ids
            })
            
            # Get model instance
            model_instance = self.model_registry.get_model(
                model.model_type,
                model.version
            )
            
            # Combine generations
            try:
                # Load source files
                sources = []
                for gen in generations:
                    if gen.file_format == "midi":
                        source = self.processor_manager.midi.load_midi(gen.file_path)
                    else:  # audio
                        source = self.processor_manager.audio.load_audio(gen.file_path)
                    sources.append(source)
                    
                # Generate combination
                output = await model_instance.predict({
                    "sources": sources,
                    **config or {}
                })
                
                # Save output file
                output_path = self.output_dir / f"{generation.resource_id}.wav"
                self.processor_manager.audio.save_audio(
                    output["audio"],
                    output_path
                )
                
                # Update generation
                file_info = self.processor_manager.audio.get_file_info(output_path)
                generation.set_file(
                    str(output_path),
                    file_info["duration"]
                )
                generation.set_outputs({
                    "file_path": str(output_path),
                    "duration": file_info["duration"]
                })
                
                # Update track
                track.add_generation(generation.resource_id)
                
                return generation
                
            except Exception as e:
                generation.delete()
                raise ProcessingError(f"Combination failed: {e}")
                
        except Exception as e:
            raise ProcessingError(f"Error in generation combination: {e}")
            
    def get_generation_info(
        self,
        generation_id: str
    ) -> Dict[str, Any]:
        """Get detailed information about a generation.
        
        Args:
            generation_id: Generation ID.
            
        Returns:
            Dictionary of generation information.
            
        Raises:
            ResourceNotFoundError: If generation not found.
        """
        # Get generation
        generation = self.resource_manager.get_resource(
            "generation",
            generation_id
        )
        
        # Get track and model
        track = self.resource_manager.get_resource(
            "track",
            generation.parent_id
        )
        model = self.resource_manager.get_resource(
            "model",
            generation.model_id
        )
        
        # Get file analysis if available
        analysis = {}
        if generation.file_path:
            try:
                file_path = Path(generation.file_path)
                if file_path.suffix == ".mid":
                    analysis = self.processor_manager.analyze_file(
                        file_path,
                        "piano_roll"
                    )
                else:
                    analysis = self.processor_manager.analyze_file(
                        file_path,
                        "spectrogram"
                    )
            except Exception as e:
                self.logger.warning(f"Error analyzing generation file: {e}")
                
        return {
            "id": generation.resource_id,
            "type": generation.generation_type,
            "status": generation.status,
            "track": {
                "id": track.resource_id,
                "name": track.name
            },
            "model": {
                "id": model.resource_id,
                "type": model.model_type,
                "version": model.version
            },
            "config": generation.config,
            "inputs": generation.inputs,
            "outputs": generation.outputs,
            "metrics": generation.metrics,
            "duration": generation.duration,
            "file_path": generation.file_path,
            "analysis": analysis,
            "created_at": generation.created_at.isoformat(),
            "updated_at": generation.updated_at.isoformat()
        }

    def quantize_project(
        self,
        project_id: str,
        target_bpm: Optional[float] = None,
        quantization_grid: str = "1/4",
        transient_sensitivity: float = 0.5
    ) -> Dict[str, Any]:
        """Quantize project to standardize timing and alignment.
        
        Args:
            project_id: Project ID.
            target_bpm: Target BPM for standardization. If None, uses project's BPM.
            quantization_grid: Grid size for quantization (e.g., "1/4", "1/8", "1/16").
            transient_sensitivity: Sensitivity for audio transient detection (0-1).
            
        Returns:
            Dictionary containing quantization results.
        """
        try:
            # Get project
            project = self.project_service.get_project(project_id)
            if not project:
                raise ResourceNotFoundError(f"Project {project_id} not found")
                
            # Get project resources
            resources = self.project_service.list_project_resources(project_id)
            
            # Initialize results
            results = {
                "project_id": project_id,
                "timestamp": datetime.utcnow().isoformat(),
                "target_bpm": target_bpm or project.get("bpm", 120),
                "quantization_grid": quantization_grid,
                "transient_sensitivity": transient_sensitivity,
                "results": {}
            }
            
            # Quantize MIDI tracks
            midi_tracks = [r for r in resources if r["type"] == "midi"]
            if midi_tracks:
                results["results"]["midi_quantization"] = self._quantize_midi_tracks(
                    project_id, midi_tracks, results["target_bpm"], quantization_grid
                )
                
            # Quantize audio tracks
            audio_tracks = [r for r in resources if r["type"] == "audio"]
            if audio_tracks:
                results["results"]["audio_quantization"] = self._quantize_audio_tracks(
                    project_id, audio_tracks, results["target_bpm"], transient_sensitivity
                )
                
            # Update project timeline
            self._update_project_timeline(project_id, results)
            
            # Save quantization results
            results_path = self.base_path / "projects" / project_id / "analysis" / "quantization_results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
                
            return results
            
        except Exception as e:
            raise ProcessingError(f"Error quantizing project: {e}")
            
    def _quantize_midi_tracks(
        self,
        project_id: str,
        midi_tracks: List[Dict[str, Any]],
        target_bpm: float,
        quantization_grid: str
    ) -> Dict[str, Any]:
        """Quantize MIDI tracks to grid.
        
        Args:
            project_id: Project ID.
            midi_tracks: List of MIDI track resources.
            target_bpm: Target BPM.
            quantization_grid: Grid size for quantization.
            
        Returns:
            Dictionary containing MIDI quantization results.
        """
        results = {
            "tracks": [],
            "overall": {
                "total_notes_quantized": 0,
                "average_offset": 0.0,
                "max_offset": 0.0
            }
        }
        
        grid_size = self._parse_grid_size(quantization_grid, target_bpm)
        
        for track in midi_tracks:
            if "notes" in track and "timing" in track:
                notes = track["notes"]
                times = track["timing"]
                quantized_times = []
                offsets = []
                
                for time in times:
                    grid_pos = round(time / grid_size) * grid_size
                    quantized_times.append(grid_pos)
                    offset = abs(time - grid_pos)
                    offsets.append(offset)
                
                track["timing"] = quantized_times
                track_path = self.base_path / "projects" / project_id / "tracks" / f"{track['id']}.json"
                
                with open(track_path, "w") as f:
                    json.dump(track, f, indent=2)
                
                track_results = {
                    "track_id": track["id"],
                    "name": track["name"],
                    "notes_quantized": len(notes),
                    "average_offset": sum(offsets) / len(offsets),
                    "max_offset": max(offsets)
                }
                results["tracks"].append(track_results)
                
                results["overall"]["total_notes_quantized"] += len(notes)
                results["overall"]["average_offset"] += track_results["average_offset"]
                results["overall"]["max_offset"] = max(
                    results["overall"]["max_offset"],
                    track_results["max_offset"]
                )
        
        num_tracks = len(midi_tracks)
        if num_tracks > 0:
            results["overall"]["average_offset"] /= num_tracks
        
        return results
        
    def _quantize_audio_tracks(
        self,
        project_id: str,
        audio_tracks: List[Dict[str, Any]],
        target_bpm: float,
        transient_sensitivity: float
    ) -> Dict[str, Any]:
        """Quantize audio tracks using transient detection.
        
        Args:
            project_id: Project ID.
            audio_tracks: List of audio track resources.
            target_bpm: Target BPM.
            transient_sensitivity: Sensitivity for transient detection.
            
        Returns:
            Dictionary containing audio quantization results.
        """
        results = {
            "tracks": [],
            "overall": {
                "total_transients_detected": 0,
                "average_offset": 0.0,
                "max_offset": 0.0
            }
        }
        
        grid_size = self._parse_grid_size("1/16", target_bpm)
        
        for track in audio_tracks:
            if "file_path" in track:
                audio_path = self.base_path / track["file_path"]
                if audio_path.exists():
                    transients = self._detect_transients(audio_path, sensitivity=transient_sensitivity)
                    quantized_transients = []
                    offsets = []
                    
                    for time in transients:
                        grid_pos = round(time / grid_size) * grid_size
                        quantized_transients.append(grid_pos)
                        offset = abs(time - grid_pos)
                        offsets.append(offset)
                    
                    track["transients"] = quantized_transients
                    track_path = self.base_path / "projects" / project_id / "tracks" / f"{track['id']}.json"
                    
                    with open(track_path, "w") as f:
                        json.dump(track, f, indent=2)
                    
                    track_results = {
                        "track_id": track["id"],
                        "name": track["name"],
                        "transients_detected": len(transients),
                        "transients_quantized": len(quantized_transients),
                        "average_offset": sum(offsets) / len(offsets) if offsets else 0,
                        "max_offset": max(offsets) if offsets else 0
                    }
                    results["tracks"].append(track_results)
                    
                    results["overall"]["total_transients_detected"] += len(transients)
                    results["overall"]["average_offset"] += track_results["average_offset"]
                    results["overall"]["max_offset"] = max(
                        results["overall"]["max_offset"],
                        track_results["max_offset"]
                    )
        
        num_tracks = len(audio_tracks)
        if num_tracks > 0:
            results["overall"]["average_offset"] /= num_tracks
        
        return results
        
    def _parse_grid_size(
        self,
        grid_size: str,
        bpm: float
    ) -> float:
        """Parse grid size string into seconds.
        
        Args:
            grid_size: Grid size string (e.g., "1/4", "1/8", "1/16").
            bpm: Beats per minute.
            
        Returns:
            Grid size in seconds.
        """
        try:
            # Parse fraction
            num, denom = map(int, grid_size.split("/"))
            
            # Calculate seconds per beat
            seconds_per_beat = 60 / bpm
            
            # Calculate grid size in seconds
            return seconds_per_beat * (num / denom)
            
        except (ValueError, ZeroDivisionError):
            return 0.25  # Default to quarter note
            
    def _detect_transients(
        self,
        audio_path: Path,
        sensitivity: float = 0.5
    ) -> List[float]:
        """Detect transients in audio file.
        
        Args:
            audio_path: Path to audio file.
            sensitivity: Detection sensitivity (0-1).
            
        Returns:
            List of transient times in seconds.
        """
        # Load audio file
        audio = self.audio_processor.load_audio(audio_path)
        
        # Calculate onset strength
        onset_env = self.audio_processor.calculate_onset_strength(audio)
        
        # Find peaks in onset strength
        peaks = self.audio_processor.find_peaks(onset_env, sensitivity)
        
        # Convert peak indices to times
        times = self.audio_processor.frames_to_time(peaks, audio)
        
        return times
        
    def _update_project_timeline(
        self,
        project_id: str,
        quantization_results: Dict[str, Any]
    ) -> None:
        """Update project timeline with quantization results.
        
        Args:
            project_id: Project ID.
            quantization_results: Quantization results.
        """
        # Get project
        project = self.project_service.get_project(project_id)
        if not project:
            return
            
        # Update project BPM
        project["bpm"] = quantization_results["target_bpm"]
        
        # Update track timings
        for track_result in quantization_results["results"].get("midi_quantization", {}).get("tracks", []):
            track_id = track_result["track_id"]
            track = self.project_service.get_project_resource(project_id, track_id)
            if track:
                track["quantized"] = True
                track["quantization_grid"] = quantization_results["quantization_grid"]
                
        for track_result in quantization_results["results"].get("audio_quantization", {}).get("tracks", []):
            track_id = track_result["track_id"]
            track = self.project_service.get_project_resource(project_id, track_id)
            if track:
                track["quantized"] = True
                track["transients"] = track_result["transients_quantized"]
                
        # Save updated project
        self.project_service.update_project(project_id, project)


# Create a global generation service instance
generation_service = GenerationService()


def get_generation_service() -> GenerationService:
    """
    Get the global generation service instance.
    
    Returns:
        GenerationService instance.
    """
    global generation_service
    return generation_service