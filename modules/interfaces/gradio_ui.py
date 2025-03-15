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
import librosa
import librosa.display
import matplotlib.pyplot as plt

from modules.services.generation_service import get_generation_service
from modules.services.model_service import get_model_service
from modules.services.output_service import get_output_service
from modules.core.visualization import create_audio_visualization, create_midi_visualization


class WizardState:
    """State management for the wizard interface."""
    def __init__(self):
        self.current_step = 0
        self.user_inputs = {}

    def next_step(self):
        self.current_step += 1
        return self.current_step

    def prev_step(self):
        self.current_step = max(0, self.current_step - 1)
        return self.current_step

    def save_input(self, key: str, value: any):
        self.user_inputs[key] = value


def create_wizard_interface() -> gr.Blocks:
    """Create a wizard interface for new users."""
    wizard_state = WizardState()
    
    with gr.Blocks() as wizard:
        gr.Markdown("# Welcome to Disco Musica!")
        gr.Markdown("Let's help you get started with AI music generation.")
        
        # Step indicators
        with gr.Row():
            step_indicators = [
                gr.Markdown(f"Step {i+1}", visible=False) 
                for i in range(4)
            ]
            step_indicators[0].visible = True
        
        # Step 1: Choose generation type
        with gr.Group() as step1:
            gr.Markdown("### Step 1: What would you like to create?")
            generation_type = gr.Radio(
                choices=["Text to Music", "Audio to Music", "MIDI to Music", "Image to Music"],
                label="Generation Type"
            )
        
        # Step 2: Input configuration
        with gr.Group(visible=False) as step2:
            gr.Markdown("### Step 2: Configure your input")
            # Dynamic inputs based on generation type
            text_config = gr.Textbox(
                label="Text Description",
                placeholder="Describe the music you want to create...",
                visible=False
            )
            audio_config = gr.Audio(
                label="Audio Input",
                visible=False
            )
            midi_config = gr.File(
                label="MIDI File",
                visible=False
            )
            image_config = gr.Image(
                label="Image Input",
                visible=False
            )
        
        # Step 3: Model and parameters
        with gr.Group(visible=False) as step3:
            gr.Markdown("### Step 3: Choose your settings")
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=["Basic", "Advanced", "Professional"],
                    value="Basic"
                )
                duration_slider = gr.Slider(
                    label="Duration (seconds)",
                    minimum=5,
                    maximum=60,
                    value=30,
                    step=5
                )
        
        # Step 4: Generate and preview
        with gr.Group(visible=False) as step4:
            gr.Markdown("### Step 4: Generate and preview your music")
            with gr.Row():
                preview_audio = gr.Audio(label="Preview")
                preview_viz = gr.Plot(label="Visualization")
        
        # Navigation buttons
        with gr.Row():
            back_btn = gr.Button("← Back", visible=False)
            next_btn = gr.Button("Next →")
            generate_btn = gr.Button("Generate!", visible=False)
        
        # Event handlers
        def update_step2_visibility(gen_type):
            wizard_state.save_input('generation_type', gen_type)
            return {
                text_config: gen_type == "Text to Music",
                audio_config: gen_type == "Audio to Music",
                midi_config: gen_type == "MIDI to Music",
                image_config: gen_type == "Image to Music"
            }
        
        generation_type.change(
            update_step2_visibility,
            inputs=[generation_type],
            outputs=[text_config, audio_config, midi_config, image_config]
        )
        
        def next_step():
            current = wizard_state.next_step()
            if current == 1:
                return {
                    step1: gr.update(visible=False),
                    step2: gr.update(visible=True),
                    back_btn: gr.update(visible=True)
                }
            elif current == 2:
                return {
                    step2: gr.update(visible=False),
                    step3: gr.update(visible=True)
                }
            elif current == 3:
                return {
                    step3: gr.update(visible=False),
                    step4: gr.update(visible=True),
                    next_btn: gr.update(visible=False),
                    generate_btn: gr.update(visible=True)
                }
        
        next_btn.click(
            next_step,
            outputs=[step1, step2, step3, step4, back_btn, next_btn, generate_btn]
        )
        
        def prev_step():
            current = wizard_state.prev_step()
            if current == 0:
                return {
                    step1: gr.update(visible=True),
                    step2: gr.update(visible=False),
                    back_btn: gr.update(visible=False)
                }
            elif current == 1:
                return {
                    step2: gr.update(visible=True),
                    step3: gr.update(visible=False)
                }
            elif current == 2:
                return {
                    step3: gr.update(visible=True),
                    step4: gr.update(visible=False),
                    next_btn: gr.update(visible=True),
                    generate_btn: gr.update(visible=False)
                }
        
        back_btn.click(
            prev_step,
            outputs=[step1, step2, step3, step4, back_btn, next_btn, generate_btn]
        )
        
        return wizard


def create_advanced_interface() -> gr.Blocks:
    """Create the advanced interface with all features."""
    # Get services
    generation_service = get_generation_service()
    model_service = get_model_service()
    output_service = get_output_service()
    
    # Get available models
    text_models = model_service.get_models_for_task("text_to_music")
    audio_models = model_service.get_models_for_task("audio_to_music")
    midi_models = model_service.get_models_for_task("midi_to_audio")
    image_models = model_service.get_models_for_task("image_to_music")
    
    with gr.Blocks(title="Disco Musica - Advanced Interface") as app:
        gr.Markdown("# Disco Musica")
        gr.Markdown("Advanced AI Music Generation Interface")
        
        with gr.Tabs():
            # Text to Music tab
            with gr.TabItem("Text to Music"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_prompt = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Describe the music you want to generate...",
                            lines=3
                        )
                        
                        with gr.Accordion("Advanced Parameters"):
                            with gr.Row():
                                text_model = gr.Dropdown(
                                    label="Model",
                                    choices=[m["id"] for m in text_models],
                                    value=text_models[0]["id"] if text_models else None
                                )
                                text_duration = gr.Slider(
                                    label="Duration (seconds)",
                                    minimum=5,
                                    maximum=120,
                                    value=30,
                                    step=5
                                )
                            
                            with gr.Row():
                                text_temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.0,
                                    maximum=1.5,
                                    value=1.0,
                                    step=0.05
                                )
                                text_seed = gr.Number(
                                    label="Seed (optional)",
                                    precision=0
                                )
                        
                        text_generate_btn = gr.Button("Generate", variant="primary")
                    
                    with gr.Column(scale=2):
                        text_output = gr.Audio(label="Generated Audio")
                        text_viz = gr.Plot(label="Visualization")
                        
                        with gr.Row():
                            text_save_btn = gr.Button("Save to Library")
                            text_download_btn = gr.Button("Download")
            
            # Audio to Music tab
            with gr.TabItem("Audio to Music"):
                with gr.Row():
                    with gr.Column(scale=3):
                        audio_input = gr.Audio(label="Input Audio")
                        audio_prompt = gr.Textbox(
                            label="Optional Text Prompt",
                            placeholder="Guide the generation with text...",
                            lines=2
                        )
                        
                        with gr.Accordion("Advanced Parameters"):
                            with gr.Row():
                                audio_model = gr.Dropdown(
                                    label="Model",
                                    choices=[m["id"] for m in audio_models],
                                    value=audio_models[0]["id"] if audio_models else None
                                )
                                audio_style = gr.Dropdown(
                                    label="Style",
                                    choices=["None", "Classical", "Jazz", "Rock", "Electronic"],
                                    value="None"
                                )
                            
                            with gr.Row():
                                audio_duration = gr.Slider(
                                    label="Duration (seconds)",
                                    minimum=5,
                                    maximum=120,
                                    value=30,
                                    step=5
                                )
                                audio_strength = gr.Slider(
                                    label="Style Strength",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.5,
                                    step=0.05
                                )
                        
                        audio_generate_btn = gr.Button("Generate", variant="primary")
                    
                    with gr.Column(scale=2):
                        audio_output = gr.Audio(label="Generated Audio")
                        audio_viz = gr.Plot(label="Visualization")
                        
                        with gr.Row():
                            audio_save_btn = gr.Button("Save to Library")
                            audio_download_btn = gr.Button("Download")
            
            # MIDI to Music tab
            with gr.TabItem("MIDI to Music"):
                with gr.Row():
                    with gr.Column(scale=3):
                        midi_input = gr.File(label="Input MIDI")
                        
                        with gr.Accordion("Advanced Parameters"):
                            with gr.Row():
                                midi_model = gr.Dropdown(
                                    label="Model",
                                    choices=[m["id"] for m in midi_models],
                                    value=midi_models[0]["id"] if midi_models else None
                                )
                                midi_instrument = gr.Dropdown(
                                    label="Primary Instrument",
                                    choices=["Piano", "Guitar", "Strings", "Synth"],
                                    value="Piano"
                                )
                            
                            with gr.Row():
                                midi_tempo = gr.Slider(
                                    label="Tempo",
                                    minimum=60,
                                    maximum=200,
                                    value=120,
                                    step=1
                                )
                                midi_reverb = gr.Slider(
                                    label="Reverb",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.3,
                                    step=0.05
                                )
                        
                        midi_generate_btn = gr.Button("Generate", variant="primary")
                    
                    with gr.Column(scale=2):
                        midi_output = gr.Audio(label="Generated Audio")
                        midi_viz = gr.Plot(label="MIDI Visualization")
                        
                        with gr.Row():
                            midi_save_btn = gr.Button("Save to Library")
                            midi_download_btn = gr.Button("Download")
            
            # Image to Music tab
            with gr.TabItem("Image to Music"):
                with gr.Row():
                    with gr.Column(scale=3):
                        image_input = gr.Image(label="Input Image")
                        image_prompt = gr.Textbox(
                            label="Optional Text Prompt",
                            placeholder="Guide the music generation with text...",
                            lines=2
                        )
                        
                        with gr.Accordion("Advanced Parameters"):
                            with gr.Row():
                                image_model = gr.Dropdown(
                                    label="Model",
                                    choices=[m["id"] for m in image_models],
                                    value=image_models[0]["id"] if image_models else None
                                )
                                image_mood = gr.Dropdown(
                                    label="Mood",
                                    choices=["Auto", "Happy", "Sad", "Energetic", "Calm"],
                                    value="Auto"
                                )
                            
                            with gr.Row():
                                image_duration = gr.Slider(
                                    label="Duration (seconds)",
                                    minimum=5,
                                    maximum=120,
                                    value=30,
                                    step=5
                                )
                                image_complexity = gr.Slider(
                                    label="Complexity",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.5,
                                    step=0.05
                                )
                        
                        image_generate_btn = gr.Button("Generate", variant="primary")
                    
                    with gr.Column(scale=2):
                        image_output = gr.Audio(label="Generated Audio")
                        image_viz = gr.Plot(label="Audio Visualization")
                        
                        with gr.Row():
                            image_save_btn = gr.Button("Save to Library")
                            image_download_btn = gr.Button("Download")
            
            # Library tab
            with gr.TabItem("Library"):
                with gr.Row():
                    with gr.Column(scale=2):
                        library_list = gr.Dataframe(
                            headers=["Name", "Type", "Duration", "Created"],
                            label="Your Generated Music"
                        )
                        with gr.Row():
                            library_refresh_btn = gr.Button("Refresh")
                            library_delete_btn = gr.Button("Delete Selected")
                    
                    with gr.Column(scale=3):
                        library_preview = gr.Audio(label="Preview")
                        library_viz = gr.Plot(label="Visualization")
                        library_metadata = gr.JSON(label="Metadata")
        
        return app


def create_ui() -> gr.Blocks:
    """Create the main UI with both wizard and advanced interfaces."""
    with gr.Blocks(title="Disco Musica") as main_app:
        # Mode selection
        with gr.Row():
            mode_select = gr.Radio(
                choices=["Wizard Mode", "Advanced Mode"],
                value="Wizard Mode",
                label="Interface Mode"
            )
        
        # Wizard interface
        wizard_interface = create_wizard_interface()
        wizard_interface.visible = True
        
        # Advanced interface
        advanced_interface = create_advanced_interface()
        advanced_interface.visible = False
        
        # Mode switching logic
        def switch_mode(mode):
            return {
                wizard_interface: gr.update(visible=mode == "Wizard Mode"),
                advanced_interface: gr.update(visible=mode == "Advanced Mode")
            }
        
        mode_select.change(
            switch_mode,
            inputs=[mode_select],
            outputs=[wizard_interface, advanced_interface]
        )
        
        return main_app


def launch_ui(debug: bool = False):
    """Launch the Disco Musica UI."""
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=debug,
        share=True
    )


if __name__ == "__main__":
    launch_ui(debug=True)