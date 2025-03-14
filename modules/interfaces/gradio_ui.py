"""
Gradio-based UI for Disco Musica.

This module provides a Gradio web interface for music generation capabilities.
"""

import os
import time
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gradio as gr

from modules.services.generation_service import get_generation_service
from modules.services.model_service import get_model_service
from modules.services.output_service import get_output_service


def create_ui():
    """
    Create and return a Gradio interface for Disco Musica.
    
    Returns:
        Gradio Blocks interface.
    """
    # Get services
    generation_service = get_generation_service()
    model_service = get_model_service()
    output_service = get_output_service()
    
    # Get available models for each task
    text_models = [model["id"] for model in model_service.get_models_for_task("text_to_music")]
    audio_models = [model["id"] for model in model_service.get_models_for_task("audio_to_music")]
    midi_models = [model["id"] for model in model_service.get_models_for_task("midi_to_audio")]
    
    default_text_model = model_service.get_default_model_for_task("text_to_music")
    default_audio_model = model_service.get_default_model_for_task("audio_to_music")
    default_midi_model = model_service.get_default_model_for_task("midi_to_audio")
    
    with gr.Blocks(title="Disco Musica") as app:
        gr.Markdown("# Disco Musica")
        gr.Markdown("An open-source multimodal AI music generation application.")
        
        with gr.Tabs():
            with gr.TabItem("Text to Music"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_prompt = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Describe the music you want to generate...",
                            lines=3
                        )
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                text_model_dropdown = gr.Dropdown(
                                    label="Model",
                                    choices=text_models,
                                    value=default_text_model
                                )
                            
                            with gr.Column(scale=1):
                                text_duration = gr.Slider(
                                    label="Duration (seconds)",
                                    minimum=5,
                                    maximum=120,
                                    value=30,
                                    step=5
                                )
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                text_temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.0,
                                    maximum=1.5,
                                    value=1.0,
                                    step=0.05
                                )
                            
                            with gr.Column(scale=1):
                                text_seed = gr.Number(
                                    label="Seed (leave empty for random)",
                                    precision=0
                                )
                        
                        text_generate_btn = gr.Button("Generate Music", variant="primary")
                    
                    with gr.Column(scale=2):
                        text_output_audio = gr.Audio(
                            label="Generated Audio",
                            type="numpy"
                        )
                        text_save_btn = gr.Button("Save to Library")
                
                # Define generation function
                def generate_from_text(prompt, model_id, duration, temperature, seed):
                    try:
                        # Convert seed to int or None
                        if seed == "" or seed is None:
                            seed_val = None
                        else:
                            seed_val = int(seed)
                        
                        # Generate music
                        result = generation_service.generate_from_text(
                            prompt=prompt,
                            model_id=model_id,
                            duration=duration,
                            temperature=temperature,
                            seed=seed_val,
                            save_output=False
                        )
                        
                        # Return audio data and sample rate
                        return (result["sample_rate"], result["audio_data"])
                    
                    except Exception as e:
                        return gr.Error(f"Generation failed: {str(e)}")
                
                # Define save function
                def save_text_generation(audio_data):
                    if audio_data is None:
                        return "No audio to save"
                    
                    try:
                        # Save to output service
                        sr, audio = audio_data
                        
                        # Create temporary metadata
                        metadata = {
                            "generation_id": f"gradio_{int(time.time())}",
                            "generation_type": "text_to_music",
                            "created_at": time.time()
                        }
                        
                        # Save to output service
                        output_info = output_service.save_audio_output(
                            audio_data=audio,
                            sample_rate=sr,
                            metadata=metadata,
                            output_format="wav",
                            visualization=True
                        )
                        
                        return f"Saved to {output_info['path']}"
                    
                    except Exception as e:
                        return f"Failed to save: {str(e)}"
                
                # Connect events
                text_generate_btn.click(
                    generate_from_text,
                    inputs=[text_prompt, text_model_dropdown, text_duration, text_temperature, text_seed],
                    outputs=text_output_audio
                )
                
                text_save_btn.click(
                    save_text_generation,
                    inputs=text_output_audio,
                    outputs=gr.Textbox()
                )
            
            with gr.TabItem("Audio to Music"):
                with gr.Row():
                    with gr.Column(scale=3):
                        audio_input = gr.Audio(
                            label="Input Audio",
                            type="filepath"
                        )
                        
                        audio_prompt = gr.Textbox(
                            label="Text Prompt (Optional)",
                            placeholder="Guide the generation process with a text description...",
                            lines=2
                        )
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                audio_model_dropdown = gr.Dropdown(
                                    label="Model",
                                    choices=audio_models,
                                    value=default_audio_model
                                )
                            
                            with gr.Column(scale=1):
                                audio_duration = gr.Slider(
                                    label="Duration (seconds)",
                                    minimum=5,
                                    maximum=120,
                                    value=30,
                                    step=5
                                )
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                audio_temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.0,
                                    maximum=1.5,
                                    value=1.0,
                                    step=0.05
                                )
                            
                            with gr.Column(scale=1):
                                audio_seed = gr.Number(
                                    label="Seed (leave empty for random)",
                                    precision=0
                                )
                        
                        audio_generate_btn = gr.Button("Generate Music", variant="primary")
                    
                    with gr.Column(scale=2):
                        audio_output_audio = gr.Audio(
                            label="Generated Audio",
                            type="numpy"
                        )
                        audio_save_btn = gr.Button("Save to Library")
                
                # Define generation function
                def generate_from_audio(audio_path, prompt, model_id, duration, temperature, seed):
                    if audio_path is None:
                        return gr.Error("Please upload an audio file")
                    
                    try:
                        # Convert seed to int or None
                        if seed == "" or seed is None:
                            seed_val = None
                        else:
                            seed_val = int(seed)
                        
                        # Generate music
                        result = generation_service.generate_from_audio(
                            audio_path=audio_path,
                            prompt=prompt,
                            model_id=model_id,
                            duration=duration,
                            temperature=temperature,
                            seed=seed_val,
                            save_output=False
                        )
                        
                        # Return audio data and sample rate
                        return (result["sample_rate"], result["audio_data"])
                    
                    except Exception as e:
                        return gr.Error(f"Generation failed: {str(e)}")
                
                # Define save function
                def save_audio_generation(audio_data):
                    if audio_data is None:
                        return "No audio to save"
                    
                    try:
                        # Save to output service
                        sr, audio = audio_data
                        
                        # Create temporary metadata
                        metadata = {
                            "generation_id": f"gradio_{int(time.time())}",
                            "generation_type": "audio_to_music",
                            "created_at": time.time()
                        }
                        
                        # Save to output service
                        output_info = output_service.save_audio_output(
                            audio_data=audio,
                            sample_rate=sr,
                            metadata=metadata,
                            output_format="wav",
                            visualization=True
                        )
                        
                        return f"Saved to {output_info['path']}"
                    
                    except Exception as e:
                        return f"Failed to save: {str(e)}"
                
                # Connect events
                audio_generate_btn.click(
                    generate_from_audio,
                    inputs=[audio_input, audio_prompt, audio_model_dropdown, audio_duration, audio_temperature, audio_seed],
                    outputs=audio_output_audio
                )
                
                audio_save_btn.click(
                    save_audio_generation,
                    inputs=audio_output_audio,
                    outputs=gr.Textbox()
                )
            
            with gr.TabItem("MIDI to Audio"):
                with gr.Row():
                    with gr.Column(scale=3):
                        midi_input = gr.File(
                            label="Input MIDI File",
                            file_types=[".mid", ".midi"]
                        )
                        
                        midi_prompt = gr.Textbox(
                            label="Text Prompt (Optional)",
                            placeholder="Guide the generation process with a text description...",
                            lines=2
                        )
                        
                        instrument_prompt = gr.Textbox(
                            label="Instrument Prompt",
                            placeholder="Describe the instruments to use (e.g., 'piano and strings', 'jazz quartet')",
                            lines=1,
                            value="piano"
                        )
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                midi_model_dropdown = gr.Dropdown(
                                    label="Model",
                                    choices=midi_models,
                                    value=default_midi_model
                                )
                            
                            with gr.Column(scale=1):
                                midi_duration = gr.Slider(
                                    label="Duration (seconds)",
                                    minimum=5,
                                    maximum=120,
                                    value=30,
                                    step=5
                                )
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                midi_temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.0,
                                    maximum=1.5,
                                    value=1.0,
                                    step=0.05
                                )
                            
                            with gr.Column(scale=1):
                                midi_seed = gr.Number(
                                    label="Seed (leave empty for random)",
                                    precision=0
                                )
                        
                        midi_generate_btn = gr.Button("Generate Music", variant="primary")
                    
                    with gr.Column(scale=2):
                        midi_output_audio = gr.Audio(
                            label="Generated Audio",
                            type="numpy"
                        )
                        midi_save_btn = gr.Button("Save to Library")
                
                # Define generation function
                def generate_from_midi(midi_path, prompt, instrument_prompt, model_id, duration, temperature, seed):
                    if midi_path is None:
                        return gr.Error("Please upload a MIDI file")
                    
                    try:
                        # Convert seed to int or None
                        if seed == "" or seed is None:
                            seed_val = None
                        else:
                            seed_val = int(seed)
                        
                        # Generate music
                        result = generation_service.generate_from_midi(
                            midi_path=midi_path,
                            prompt=prompt,
                            instrument_prompt=instrument_prompt,
                            model_id=model_id,
                            duration=duration,
                            temperature=temperature,
                            seed=seed_val,
                            save_output=False
                        )
                        
                        # Return audio data and sample rate
                        return (result["sample_rate"], result["audio_data"])
                    
                    except Exception as e:
                        return gr.Error(f"Generation failed: {str(e)}")
                
                # Define save function
                def save_midi_generation(audio_data):
                    if audio_data is None:
                        return "No audio to save"
                    
                    try:
                        # Save to output service
                        sr, audio = audio_data
                        
                        # Create temporary metadata
                        metadata = {
                            "generation_id": f"gradio_{int(time.time())}",
                            "generation_type": "midi_to_audio",
                            "created_at": time.time()
                        }
                        
                        # Save to output service
                        output_info = output_service.save_audio_output(
                            audio_data=audio,
                            sample_rate=sr,
                            metadata=metadata,
                            output_format="wav",
                            visualization=True
                        )
                        
                        return f"Saved to {output_info['path']}"
                    
                    except Exception as e:
                        return f"Failed to save: {str(e)}"
                
                # Connect events
                midi_generate_btn.click(
                    generate_from_midi,
                    inputs=[midi_input, midi_prompt, instrument_prompt, midi_model_dropdown, midi_duration, midi_temperature, midi_seed],
                    outputs=midi_output_audio
                )
                
                midi_save_btn.click(
                    save_midi_generation,
                    inputs=midi_output_audio,
                    outputs=gr.Textbox()
                )
        
        # Add a tab for library access
        with gr.TabItem("Library"):
            gr.Markdown("### Generated Music Library")
            
            with gr.Row():
                refresh_library_btn = gr.Button("Refresh Library")
                
            with gr.Row():
                with gr.Column():
                    library_output = gr.Dataframe(
                        headers=["ID", "Type", "Timestamp", "Path"],
                        label="Generated Outputs"
                    )
            
            with gr.Row():
                with gr.Column():
                    selected_output_id = gr.Textbox(label="Selected Output ID")
                    load_output_btn = gr.Button("Load Selected Output")
                
                with gr.Column():
                    library_audio = gr.Audio(label="Selected Output")
            
            # Library functions
            def refresh_library():
                outputs = output_service.list_outputs()
                
                # Format for dataframe
                rows = []
                for output in outputs:
                    rows.append([
                        output.get("id", ""),
                        output.get("type", ""),
                        output.get("timestamp", ""),
                        output.get("path", "")
                    ])
                
                return rows
            
            def load_output(output_id):
                if not output_id:
                    return gr.Error("Please select an output ID")
                
                try:
                    output = output_service.load_output(output_id)
                    
                    if output is None:
                        return gr.Error(f"Output {output_id} not found")
                    
                    if "audio_data" not in output:
                        return gr.Error(f"No audio data found for output {output_id}")
                    
                    # Return audio data
                    return (output.get("sample_rate", 44100), output["audio_data"])
                
                except Exception as e:
                    return gr.Error(f"Failed to load output: {str(e)}")
            
            # Connect events
            refresh_library_btn.click(
                refresh_library,
                outputs=library_output
            )
            
            load_output_btn.click(
                load_output,
                inputs=selected_output_id,
                outputs=library_audio
            )
    
    return app


def launch_ui(debug=False):
    """
    Create and launch the Gradio web interface.
    
    Args:
        debug: Whether to run in debug mode.
    """
    app = create_ui()
    app.launch(share=False, inbrowser=True, debug=debug)


if __name__ == "__main__":
    launch_ui(debug=True)