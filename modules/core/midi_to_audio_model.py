"""
MIDI-to-Audio model module for Disco Musica.

This module provides a wrapper for MIDI-to-audio generation models,
with support for rendering MIDI into realistic audio performances.
"""

import os
import time
import datetime
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from modules.core.base_model import BaseModel, PretrainedModelMixin, TorchModelMixin
from modules.core.config import config
from modules.core.audio_processor import AudioProcessor
from modules.core.midi_processor import MIDIProcessor


class MIDIToAudioModel(BaseModel, PretrainedModelMixin, TorchModelMixin):
    """
    Model for generating audio from MIDI inputs.
    
    This class wraps MIDI-to-audio models, providing a consistent
    interface for loading, saving, and generating audio from MIDI files.
    """
    
    def __init__(
        self, 
        model_name: str = "facebook/musicgen-melody", 
        model_type: str = "pretrained"
    ):
        """
        Initialize the MIDIToAudioModel.
        
        Args:
            model_name: Name of the model. Default is "facebook/musicgen-melody".
            model_type: Type of model ('pretrained' or 'finetuned').
        """
        super().__init__(model_name, model_type)
        
        self.metadata.update({
            "capabilities": ["midi_to_audio"],
            "version": "0.1.0",
            "creation_date": datetime.datetime.now().isoformat(),
            "parameters": {
                "max_duration": config.get("inference", "max_generation_length", 30.0),
                "sample_rate": config.get("audio", "sample_rate", 44100),
                "temperature": config.get("inference", "temperature", 1.0),
                "top_k": config.get("inference", "top_k", 250),
                "top_p": config.get("inference", "top_p", 0.0),
                "cfg_coef": config.get("inference", "classifier_free_guidance", 3.0)
            }
        })
        
        # Flag to track model loading status
        self._is_loaded = False
        
        # MusicGen model and processor
        self.model = None
        self.processor = None
        
        # Audio and MIDI processors
        self.audio_processor = AudioProcessor()
        self.midi_processor = MIDIProcessor()
    
    def load(self) -> None:
        """
        Load the model from disk or download if necessary.
        """
        if self._is_loaded:
            return
            
        model_path = self.get_model_path()
        
        try:
            # Try to import transformers
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            
            # Check if model exists locally
            if model_path.exists() and (model_path / "config.json").exists():
                print(f"Loading model from {model_path}")
                self.model = MusicgenForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                ).to(self.device)
                self.processor = AutoProcessor.from_pretrained(model_path)
            else:
                # Download from Hugging Face Hub
                print(f"Downloading model {self.model_name} from Hugging Face Hub")
                self.model = MusicgenForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                ).to(self.device)
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                
                # Save model locally
                os.makedirs(model_path, exist_ok=True)
                self.model.save_pretrained(model_path)
                self.processor.save_pretrained(model_path)
            
            # Update metadata
            self.metadata.update({
                "last_modified": datetime.datetime.now().isoformat(),
                "parameters": {
                    **self.metadata["parameters"],
                    "model_size": self.get_model_size()
                }
            })
            self.save_metadata()
            
            self._is_loaded = True
            print(f"Model {self.model_name} loaded successfully")
        
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def save(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save the model to disk.
        
        Args:
            output_path: Path to save the model. If None, a default path based on 
                         model name and type will be used.
        
        Returns:
            Path where the model was saved.
        """
        if not self._is_loaded or self.model is None:
            raise ValueError("Model must be loaded before saving")
        
        if output_path is None:
            output_path = self.get_model_path()
        else:
            output_path = Path(output_path)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Save model and processor
            self.model.save_pretrained(output_path)
            if self.processor is not None:
                self.processor.save_pretrained(output_path)
            
            # Update and save metadata
            self.metadata.update({
                "last_modified": datetime.datetime.now().isoformat()
            })
            self.save_metadata(output_path / "metadata.json")
            
            print(f"Model saved to {output_path}")
            return output_path
        
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
    
    def generate(
        self, 
        midi_input: Union[str, Path],
        prompt: Optional[str] = None,
        instrument_prompt: Optional[str] = None,
        duration: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        cfg_coef: Optional[float] = None,
        return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate audio from a MIDI input.
        
        Args:
            midi_input: Path to the MIDI file.
            prompt: Optional text prompt to guide generation.
            instrument_prompt: Optional prompt describing the instrumentation.
            duration: Duration of the generated audio in seconds.
            temperature: Sampling temperature (higher = more random).
            top_k: Number of top tokens to consider (0 = disabled).
            top_p: Nucleus sampling probability threshold (0 = disabled).
            cfg_coef: Classifier-free guidance coefficient (higher = more adherence to prompt).
            return_tensor: Whether to return a torch.Tensor (True) or numpy array (False).
            
        Returns:
            Generated audio as a numpy array or torch.Tensor.
        """
        if not self._is_loaded:
            self.load()
        
        # Use default values from metadata if not provided
        duration = duration or self.metadata["parameters"].get("max_duration", 30.0)
        temperature = temperature or self.metadata["parameters"].get("temperature", 1.0)
        top_k = top_k or self.metadata["parameters"].get("top_k", 250)
        top_p = top_p or self.metadata["parameters"].get("top_p", 0.0)
        cfg_coef = cfg_coef or self.metadata["parameters"].get("cfg_coef", 3.0)
        
        try:
            # Process MIDI file to extract musical features
            midi_score = self.midi_processor.load_midi(midi_input)
            
            # Extract key, tempo, and time signature
            key = self.midi_processor.extract_key(midi_score) or "unknown key"
            tempo = self.midi_processor.extract_tempo(midi_score) or 120
            time_sig = self.midi_processor.extract_time_signature(midi_score) or "4/4"
            
            # Extract melody notes
            melody = self.midi_processor.extract_melody(midi_score)
            
            # Create a descriptive prompt if not provided
            if prompt is None:
                # Get the MIDI filename without extension
                midi_name = Path(midi_input).stem.replace("_", " ").replace("-", " ")
                prompt = f"A {key} {time_sig} piece at {tempo} BPM called '{midi_name}'"
                
                if instrument_prompt:
                    prompt += f" performed on {instrument_prompt}"
            
            # For now, we'll use a text-to-music approach with a descriptive prompt
            # In a real implementation, we'd want to condition on the actual MIDI data
            
            # Prepare inputs
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Calculate max_new_tokens based on duration and model's audio_config
            sample_rate = self.model.config.audio_encoder.sampling_rate
            chunk_length = self.model.config.audio_encoder.chunk_length
            max_new_tokens = int(duration * sample_rate / chunk_length)
            
            # Generate audio
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
            }
            
            if top_k > 0:
                generation_kwargs["top_k"] = top_k
                
            if top_p > 0:
                generation_kwargs["top_p"] = top_p
                
            if cfg_coef > 0:
                generation_kwargs["guidance_scale"] = cfg_coef
            
            # Start generation
            start_time = time.time()
            
            with torch.inference_mode():
                generated_audio = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            generation_time = time.time() - start_time
            print(f"Generation took {generation_time:.2f} seconds")
            
            # Convert to audio waveforms
            audio_values = self.processor.batch_decode(generated_audio, return_tensors="pt")
            
            # Return the generated audio
            if return_tensor:
                return audio_values[0]  # Return first batch item as tensor
            else:
                return audio_values[0].cpu().numpy()  # Return first batch item as numpy array
                
        except Exception as e:
            raise RuntimeError(f"MIDI-to-audio generation failed: {e}")
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded.
        
        Returns:
            True if the model is loaded, False otherwise.
        """
        return self._is_loaded and self.model is not None
    
    def render_with_style(
        self,
        midi_input: Union[str, Path],
        style_prompt: str,
        duration: Optional[float] = None,
        temperature: Optional[float] = None,
        return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Render a MIDI file in a specific style.
        
        Args:
            midi_input: Path to the MIDI file.
            style_prompt: Text prompt describing the desired style (e.g., "jazz piano", "orchestral").
            duration: Duration of the generated audio in seconds.
            temperature: Sampling temperature (higher = more random).
            return_tensor: Whether to return a torch.Tensor (True) or numpy array (False).
            
        Returns:
            Generated audio as a numpy array or torch.Tensor.
        """
        return self.generate(
            midi_input=midi_input,
            instrument_prompt=style_prompt,
            duration=duration,
            temperature=temperature,
            return_tensor=return_tensor
        )
    
    def harmonize(
        self,
        midi_input: Union[str, Path],
        style_prompt: str = "full band arrangement",
        complexity: float = 0.7,
        duration: Optional[float] = None,
        return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Harmonize a melody from a MIDI file.
        
        Args:
            midi_input: Path to the MIDI file.
            style_prompt: Text prompt describing the desired style.
            complexity: Harmonic complexity (0.0 to 1.0).
            duration: Duration of the generated audio in seconds.
            return_tensor: Whether to return a torch.Tensor (True) or numpy array (False).
            
        Returns:
            Harmonized audio as a numpy array or torch.Tensor.
        """
        # Process MIDI file to extract melody
        midi_score = self.midi_processor.load_midi(midi_input)
        
        # Extract key and tempo
        key = self.midi_processor.extract_key(midi_score) or "unknown key"
        tempo = self.midi_processor.extract_tempo(midi_score) or 120
        
        # Create harmonization prompt
        complexity_term = "complex" if complexity > 0.5 else "simple"
        harmonization_prompt = f"Create a {complexity_term} {style_prompt} in {key} at {tempo} BPM, harmonizing the melody"
        
        # Apply higher temperature for more creative harmonization
        temp = 0.9 if complexity > 0.5 else 0.7
        
        return self.generate(
            midi_input=midi_input,
            prompt=harmonization_prompt,
            duration=duration,
            temperature=temp,
            return_tensor=return_tensor
        )