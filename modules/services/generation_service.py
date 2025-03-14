"""
Generation Service module for Disco Musica.

This module provides services for music generation across different modalities.
"""

import os
import time
import datetime
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from modules.core.config import config
from modules.core.audio_processor import AudioProcessor
from modules.services.model_service import get_model_service
from modules.core.text_to_music_model import TextToMusicModel


class GenerationService:
    """
    Service for music generation across different modalities.
    
    This class provides a unified interface for generating music from text,
    audio, MIDI, and image inputs, handling the model selection, generation,
    and output processing.
    """
    
    def __init__(self):
        """
        Initialize the GenerationService.
        """
        self.model_service = get_model_service()
        self.audio_processor = AudioProcessor()
        
        # Output directory
        self.output_dir = Path(config.get("paths", "output_dir", "outputs"))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generation history
        self.history = {}
    
    def generate_from_text(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        duration: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        save_output: bool = True,
        output_format: str = "wav"
    ) -> Dict:
        """
        Generate music from a text prompt.
        
        Args:
            prompt: Text prompt describing the music to generate.
            model_id: ID of the model to use. If None, the default model is used.
            duration: Duration of the generated audio in seconds.
            temperature: Sampling temperature (higher = more random).
            top_k: Number of top tokens to consider (0 = disabled).
            top_p: Nucleus sampling probability threshold (0 = disabled).
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
            model_id = self.model_service.get_default_model_for_task("text_to_music")
            if model_id is None:
                raise ValueError("No default model found for text-to-music generation")
        
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
                prompt=prompt,
                duration=duration,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Get model parameters
            sample_rate = model.metadata["parameters"].get("sample_rate", 44100)
            
            # Save output if requested
            output_path = None
            if save_output:
                # Create output directory
                output_subdir = self.output_dir / "text_to_music"
                os.makedirs(output_subdir, exist_ok=True)
                
                # Generate filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"text_to_music_{timestamp}.{output_format}"
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
                "audio_data": audio_data,
                "sample_rate": sample_rate,
                "duration": len(audio_data) / sample_rate,
                "output_path": str(output_path) if output_path else None,
                "generation_time": generation_time,
                "parameters": {
                    "duration": duration,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "seed": seed
                },
                "analysis": audio_analysis
            }
            
            # Save to history
            self.history[generation_id] = {
                "type": "text_to_music",
                "timestamp": datetime.datetime.now().isoformat(),
                "model_id": model_id,
                "prompt": prompt,
                "output_path": str(output_path) if output_path else None,
                "parameters": result["parameters"]
            }
            
            return result
        
        except Exception as e:
            print(f"Error in text-to-music generation: {e}")
            raise
    
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