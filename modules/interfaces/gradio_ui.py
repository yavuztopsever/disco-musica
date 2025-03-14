"""
Gradio UI interface for Disco Musica.

This module provides a web-based user interface for the Disco Musica application
using the Gradio library.
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import gradio as gr

from modules.core.config import config
from modules.services.model_service import get_model_service
from modules.services.generation_service import get_generation_service
from modules.services.output_service import get_output_service


class GradioInterface:
    """
    Gradio-based user interface for Disco Musica.
    
    This class provides a web-based interface for interacting with the Disco Musica
    application, including music generation and model fine-tuning.
    """
    
    def __init__(self):
        """
        Initialize the GradioInterface.
        """
        self.model_service = get_model_service()
        self.generation_service = get_generation_service()
        self.output_service = get_output_service()
        
        # UI theme
        self.theme = gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
        ).set(
            body_background_fill="*neutral_50",
            block_background_fill="*neutral_100",
            block_label_background_fill="*primary_100",
            block_border_color="*neutral_200",
        )
        
        # Initialize interface
        self.interface = None
    
    def text_to_music_interface(self) -> List[gr.Component]:
        """
        Create the text-to-music interface components.
        
        Returns:
            List of Gradio components.
        """
        # Get available models
        models = self.model_service.get_models_for_task("text_to_music")
        model_choices = [
            model["id"] for model in models
        ] if models else ["facebook/musicgen-small"]
        
        # Text input
        text_input = gr.Textbox(
            label="Text Prompt",
            placeholder="Describe the music you want to generate...",
            lines=3
        )
        
        # Model selection
        model_dropdown = gr.Dropdown(
            label="Model",
            choices=model_choices,
            value=model_choices[0] if model_choices else None
        )
        
        # Generation parameters
        with gr.Row():
            duration_slider = gr.Slider(
                label="Duration (seconds)",
                minimum=5,
                maximum=60,
                value=10,
                step=5
            )
            
            temperature_slider = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1
            )
        
        with gr.Row():
            top_k_slider = gr.Slider(
                label="Top-K",
                minimum=0,
                maximum=1000,
                value=250,
                step=50
            )
            
            top_p_slider = gr.Slider(
                label="Top-P",
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.05
            )
        
        seed_number = gr.Number(
            label="Seed (0 for random)",
            value=0,
            precision=0
        )
        
        # Generate button
        generate_button = gr.Button("Generate Music", variant="primary")
        
        # Output components
        audio_output = gr.Audio(
            label="Generated Music",
            type="numpy"
        )
        
        waveform_output = gr.Plot(
            label="Waveform"
        )
        
        generation_info = gr.JSON(
            label="Generation Info"
        )
        
        # Save output components
        with gr.Row():
            save_button = gr.Button("Save to Library")
            download_button = gr.Button("Download")
        
        # Function to generate music
        def generate_music(
            prompt, model_id, duration, temperature, 
            top_k, top_p, seed
        ):
            try:
                # Set seed if specified
                if seed <= 0:
                    seed = None
                
                # Generate music
                result = self.generation_service.generate_from_text(
                    prompt=prompt,
                    model_id=model_id,
                    duration=duration,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                    save_output=False
                )
                
                # Create waveform plot
                import matplotlib.pyplot as plt
                import librosa.display
                
                fig = plt.figure(figsize=(10, 4))
                librosa.display.waveshow(
                    result["audio_data"], 
                    sr=result["sample_rate"]
                )
                plt.title("Waveform")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.tight_layout()
                
                # Return results
                output_info = {
                    "duration": result["duration"],
                    "generation_time": result["generation_time"],
                    "model": model_id,
                    "parameters": {
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "seed": seed
                    }
                }
                
                return (
                    (result["audio_data"], result["sample_rate"]),
                    fig,
                    output_info
                )
            
            except Exception as e:
                print(f"Error generating music: {e}")
                return None, None, {"error": str(e)}
        
        # Connect components
        generate_button.click(
            generate_music,
            inputs=[
                text_input, model_dropdown, duration_slider,
                temperature_slider, top_k_slider, top_p_slider,
                seed_number
            ],
            outputs=[audio_output, waveform_output, generation_info]
        )
        
        # Function to save output
        def save_output(audio, generation_info, prompt):
            try:
                if audio is None or audio[0] is None:
                    return {"status": "error", "message": "No audio to save"}
                
                audio_data, sample_rate = audio
                
                # Create metadata
                metadata = {
                    "prompt": prompt,
                    "model_id": generation_info.get("model", "unknown"),
                    "parameters": generation_info.get("parameters", {}),
                    "tags": ["text-to-music"]
                }
                
                # Save to library
                output_info = self.output_service.save_audio_output(
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    metadata=metadata,
                    output_format="wav",
                    visualization=True
                )
                
                return {"status": "success", "output_id": output_info["id"]}
            
            except Exception as e:
                print(f"Error saving output: {e}")
                return {"status": "error", "message": str(e)}
        
        # Connect save button
        save_button.click(
            save_output,
            inputs=[audio_output, generation_info, text_input],
            outputs=[gr.JSON(label="Save Result")]
        )
        
        # Function to download output
        def download_output(audio):
            try:
                if audio is None or audio[0] is None:
                    return None
                
                audio_data, sample_rate = audio
                
                # Create temporary file
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, "generated_music.wav")
                
                # Save audio to temporary file
                import soundfile as sf
                sf.write(temp_path, audio_data, sample_rate)
                
                return temp_path
            
            except Exception as e:
                print(f"Error preparing download: {e}")
                return None
        
        # Connect download button
        download_button.click(
            download_output,
            inputs=[audio_output],
            outputs=[gr.File(label="Download Generated Music")]
        )
        
        return [
            text_input, model_dropdown, duration_slider,
            temperature_slider, top_k_slider, top_p_slider,
            seed_number, generate_button, audio_output,
            waveform_output, generation_info, save_button,
            download_button
        ]
    
    def audio_to_music_interface(self) -> List[gr.Component]:
        """
        Create the audio-to-music interface components.
        
        Returns:
            List of Gradio components.
        """
        # Get available models
        models = self.model_service.get_models_for_task("audio_to_music")
        model_choices = [
            model["id"] for model in models
        ] if models else ["facebook/musicgen-melody"]
        
        # Audio input
        audio_input = gr.Audio(
            label="Input Audio",
            type="filepath"
        )
        
        # Text prompt (optional)
        text_input = gr.Textbox(
            label="Text Prompt (Optional)",
            placeholder="Describe how to transform the input audio...",
            lines=2
        )
        
        # Model selection
        model_dropdown = gr.Dropdown(
            label="Model",
            choices=model_choices,
            value=model_choices[0] if model_choices else None
        )
        
        # Generation parameters
        with gr.Row():
            duration_slider = gr.Slider(
                label="Duration (seconds)",
                minimum=5,
                maximum=60,
                value=10,
                step=5
            )
            
            temperature_slider = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1
            )
        
        seed_number = gr.Number(
            label="Seed (0 for random)",
            value=0,
            precision=0
        )
        
        # Generate button
        generate_button = gr.Button("Generate Music", variant="primary")
        
        # Output components
        audio_output = gr.Audio(
            label="Generated Music",
            type="numpy"
        )
        
        waveform_output = gr.Plot(
            label="Waveform"
        )
        
        generation_info = gr.JSON(
            label="Generation Info"
        )
        
        # Save output components
        with gr.Row():
            save_button = gr.Button("Save to Library")
            download_button = gr.Button("Download")
        
        # Function to generate music from audio
        def generate_music_from_audio(
            audio_path, prompt, model_id, duration, temperature, seed
        ):
            try:
                if audio_path is None:
                    return None, None, {"error": "No audio input provided"}
                
                # Set seed if specified
                if seed <= 0:
                    seed = None
                
                # Generate music
                result = self.generation_service.generate_from_audio(
                    audio_path=audio_path,
                    prompt=prompt,
                    model_id=model_id,
                    duration=duration,
                    temperature=temperature,
                    seed=seed,
                    save_output=False
                )
                
                # Create waveform plot
                import matplotlib.pyplot as plt
                import librosa.display
                
                fig = plt.figure(figsize=(10, 4))
                librosa.display.waveshow(
                    result["audio_data"], 
                    sr=result["sample_rate"]
                )
                plt.title("Waveform")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.tight_layout()
                
                # Return results
                output_info = {
                    "duration": result["duration"],
                    "generation_time": result["generation_time"],
                    "model": model_id,
                    "parameters": {
                        "temperature": temperature,
                        "seed": seed
                    }
                }
                
                return (
                    (result["audio_data"], result["sample_rate"]),
                    fig,
                    output_info
                )
            
            except Exception as e:
                print(f"Error generating music from audio: {e}")
                return None, None, {"error": str(e)}
        
        # Connect components
        generate_button.click(
            generate_music_from_audio,
            inputs=[
                audio_input, text_input, model_dropdown,
                duration_slider, temperature_slider, seed_number
            ],
            outputs=[audio_output, waveform_output, generation_info]
        )
        
        # Function to save output (similar to text-to-music but with different metadata)
        def save_output(audio, generation_info, audio_path, prompt):
            try:
                if audio is None or audio[0] is None:
                    return {"status": "error", "message": "No audio to save"}
                
                audio_data, sample_rate = audio
                
                # Create metadata
                metadata = {
                    "input_audio": audio_path,
                    "prompt": prompt,
                    "model_id": generation_info.get("model", "unknown"),
                    "parameters": generation_info.get("parameters", {}),
                    "tags": ["audio-to-music"]
                }
                
                # Save to library
                output_info = self.output_service.save_audio_output(
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    metadata=metadata,
                    output_format="wav",
                    visualization=True
                )
                
                return {"status": "success", "output_id": output_info["id"]}
            
            except Exception as e:
                print(f"Error saving output: {e}")
                return {"status": "error", "message": str(e)}
        
        # Connect save button
        save_button.click(
            save_output,
            inputs=[audio_output, generation_info, audio_input, text_input],
            outputs=[gr.JSON(label="Save Result")]
        )
        
        # Connect download button (reuse the download function from text-to-music)
        download_button.click(
            download_output,
            inputs=[audio_output],
            outputs=[gr.File(label="Download Generated Music")]
        )
        
        return [
            audio_input, text_input, model_dropdown, duration_slider,
            temperature_slider, seed_number, generate_button, audio_output,
            waveform_output, generation_info, save_button, download_button
        ]
    
    def library_interface(self) -> List[gr.Component]:
        """
        Create the library interface components.
        
        Returns:
            List of Gradio components.
        """
        # Library tabs
        with gr.Row():
            num_outputs = gr.Number(label="Number of Items", value=0, interactive=False)
            refresh_button = gr.Button("Refresh Library")
        
        # Outputs table
        outputs_table = gr.Dataframe(
            headers=["ID", "Type", "Created", "Tags"],
            label="Generated Outputs",
            interactive=False
        )
        
        # Selected output display
        with gr.Row():
            with gr.Column(scale=1):
                output_id = gr.Textbox(label="Output ID", interactive=False)
                output_type = gr.Textbox(label="Type", interactive=False)
                output_tags = gr.Textbox(label="Tags", interactive=False)
                
                # Tag management
                new_tag = gr.Textbox(label="Add Tag", placeholder="Enter a new tag...")
                add_tag_button = gr.Button("Add Tag")
            
            with gr.Column(scale=2):
                audio_player = gr.Audio(label="Audio", type="numpy")
                waveform_display = gr.Plot(label="Waveform")
                metadata_display = gr.JSON(label="Metadata")
        
        # Action buttons
        with gr.Row():
            delete_button = gr.Button("Delete", variant="stop")
            export_button = gr.Button("Export", variant="secondary")
            create_variation_button = gr.Button("Create Variation", variant="secondary")
        
        # Function to refresh library
        def refresh_library():
            try:
                outputs = self.output_service.list_outputs()
                
                # Format for table display
                table_data = []
                for output in outputs:
                    created_at = output.get("timestamp", "").split("T")[0]
                    tags = ", ".join(output.get("tags", []))
                    table_data.append([
                        output["id"],
                        output["type"],
                        created_at,
                        tags
                    ])
                
                return len(outputs), table_data
            
            except Exception as e:
                print(f"Error refreshing library: {e}")
                return 0, []
        
        # Connect refresh button
        refresh_button.click(
            refresh_library,
            inputs=[],
            outputs=[num_outputs, outputs_table]
        )
        
        # Function to load output details
        def load_output_details(selected_rows, outputs_table):
            try:
                if not selected_rows or len(selected_rows) == 0:
                    return None, None, None, None, None, None
                
                # Get output ID from selected row
                output_id = outputs_table[selected_rows[0]][0]
                
                # Load output
                output = self.output_service.load_output(output_id)
                if not output:
                    return None, None, None, None, None, None
                
                # Get output details
                output_type = output.get("type", "unknown")
                tags = ", ".join(output.get("tags", []))
                
                # Get audio data if available
                audio_data = None
                if "audio_data" in output and "sample_rate" in output:
                    audio_data = (output["audio_data"], output["sample_rate"])
                
                # Create waveform plot if audio data is available
                waveform_plot = None
                if audio_data:
                    import matplotlib.pyplot as plt
                    import librosa.display
                    
                    fig = plt.figure(figsize=(10, 4))
                    librosa.display.waveshow(
                        audio_data[0], 
                        sr=audio_data[1]
                    )
                    plt.title("Waveform")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Amplitude")
                    plt.tight_layout()
                    
                    waveform_plot = fig
                
                # Return output details
                return output_id, output_type, tags, audio_data, waveform_plot, output
            
            except Exception as e:
                print(f"Error loading output details: {e}")
                return None, None, None, None, None, None
        
        # Connect outputs table
        outputs_table.select(
            load_output_details,
            inputs=[outputs_table.select_index, outputs_table],
            outputs=[output_id, output_type, output_tags, audio_player, waveform_display, metadata_display]
        )
        
        # Function to add tag
        def add_tag(output_id_value, new_tag_value):
            try:
                if not output_id_value or not new_tag_value:
                    return "No output selected or tag provided", None
                
                # Add tag
                success = self.output_service.tag_output(
                    output_id=output_id_value,
                    tags=[new_tag_value]
                )
                
                if success:
                    # Refresh table
                    num_outputs_value, table_data = refresh_library()
                    
                    # Reload output details
                    for i, row in enumerate(table_data):
                        if row[0] == output_id_value:
                            output_id_value, output_type_value, tags_value, audio_data, waveform_plot, metadata = load_output_details(
                                [i], table_data
                            )
                            return "", tags_value
                
                return "Failed to add tag", None
            
            except Exception as e:
                print(f"Error adding tag: {e}")
                return f"Error: {str(e)}", None
        
        # Connect add tag button
        add_tag_button.click(
            add_tag,
            inputs=[output_id, new_tag],
            outputs=[gr.Textbox(visible=False), output_tags]
        )
        
        # Function to delete output
        def delete_output(output_id_value):
            try:
                if not output_id_value:
                    return "No output selected", None, None
                
                # Delete output
                success = self.output_service.delete_output(
                    output_id=output_id_value
                )
                
                if success:
                    # Refresh table
                    num_outputs_value, table_data = refresh_library()
                    
                    return f"Deleted output {output_id_value}", num_outputs_value, table_data
                
                return f"Failed to delete output {output_id_value}", None, None
            
            except Exception as e:
                print(f"Error deleting output: {e}")
                return f"Error: {str(e)}", None, None
        
        # Connect delete button
        delete_button.click(
            delete_output,
            inputs=[output_id],
            outputs=[gr.Textbox(visible=False), num_outputs, outputs_table]
        )
        
        # Function to export output
        def export_output(output_id_value):
            try:
                if not output_id_value:
                    return None
                
                # Get output info
                output = self.output_service.load_output(output_id_value)
                if not output:
                    return None
                
                # Get file path
                file_path = output.get("audio_path") or output.get("midi_path")
                if not file_path:
                    return None
                
                return file_path
            
            except Exception as e:
                print(f"Error exporting output: {e}")
                return None
        
        # Connect export button
        export_button.click(
            export_output,
            inputs=[output_id],
            outputs=[gr.File(label="Export Output")]
        )
        
        # Load library on start
        num_outputs_value, table_data = refresh_library()
        
        return [
            num_outputs, refresh_button, outputs_table,
            output_id, output_type, output_tags, new_tag,
            add_tag_button, audio_player, waveform_display,
            metadata_display, delete_button, export_button,
            create_variation_button
        ]
    
    def model_management_interface(self) -> List[gr.Component]:
        """
        Create the model management interface components.
        
        Returns:
            List of Gradio components.
        """
        # Model search
        with gr.Row():
            search_text = gr.Textbox(
                label="Search Models",
                placeholder="Enter search terms..."
            )
            
            task_dropdown = gr.Dropdown(
                label="Task",
                choices=[
                    "text_to_music",
                    "audio_to_music",
                    "midi_to_audio",
                    "image_to_music"
                ],
                value="text_to_music"
            )
            
            search_button = gr.Button("Search")
        
        # Model list
        models_table = gr.Dataframe(
            headers=["ID", "Name", "Type", "Status"],
            label="Available Models",
            interactive=False
        )
        
        # Model details
        with gr.Row():
            with gr.Column(scale=1):
                model_id = gr.Textbox(label="Model ID", interactive=False)
                model_name = gr.Textbox(label="Name", interactive=False)
                model_type = gr.Textbox(label="Type", interactive=False)
                model_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=2):
                model_description = gr.Textbox(
                    label="Description",
                    lines=3,
                    interactive=False
                )
                model_capabilities = gr.Textbox(
                    label="Capabilities",
                    interactive=False
                )
                model_details = gr.JSON(
                    label="Details",
                    interactive=False
                )
        
        # Action buttons
        with gr.Row():
            download_model_button = gr.Button("Download", variant="primary")
            set_default_button = gr.Button("Set as Default", variant="secondary")
        
        # Function to search models
        def search_models(search_query, task):
            try:
                # Search for models
                models = []
                
                # First check local models
                local_models = self.model_service.get_models_for_task(task)
                for model in local_models:
                    if not search_query or search_query.lower() in model["id"].lower():
                        models.append(model)
                
                # Then search Hugging Face if search query is provided
                if search_query:
                    hf_models = self.model_service.search_huggingface_models(
                        query=search_query,
                        task_type=task,
                        limit=10
                    )
                    
                    for model in hf_models:
                        # Check if model is already in the list
                        if not any(m["id"] == model["id"] for m in models):
                            models.append(model)
                
                # Format for table display
                table_data = []
                for model in models:
                    # Check if the model is available locally
                    status = "Available Locally" if self.model_service.is_model_available_locally(model["id"]) else "Not Downloaded"
                    
                    # Add row
                    table_data.append([
                        model["id"],
                        model.get("name", model["id"].split("/")[-1]),
                        model.get("model_class", "Unknown"),
                        status
                    ])
                
                return table_data
            
            except Exception as e:
                print(f"Error searching models: {e}")
                return []
        
        # Connect search button
        search_button.click(
            search_models,
            inputs=[search_text, task_dropdown],
            outputs=[models_table]
        )
        
        # Function to load model details
        def load_model_details(selected_rows, models_table):
            try:
                if not selected_rows or len(selected_rows) == 0:
                    return None, None, None, None, None, None, None
                
                # Get model ID from selected row
                model_id_value = models_table[selected_rows[0]][0]
                
                # Get model info
                model_info = self.model_service.get_model_info(model_id_value)
                if not model_info:
                    return None, None, None, None, None, None, None
                
                # Get model details
                model_name_value = model_info.get("name", model_id_value.split("/")[-1])
                model_type_value = model_info.get("model_class", "Unknown")
                model_status_value = "Available Locally" if self.model_service.is_model_available_locally(model_id_value) else "Not Downloaded"
                model_description_value = model_info.get("description", "No description available")
                model_capabilities_value = ", ".join(model_info.get("capabilities", []))
                
                # Return model details
                return (
                    model_id_value,
                    model_name_value,
                    model_type_value,
                    model_status_value,
                    model_description_value,
                    model_capabilities_value,
                    model_info
                )
            
            except Exception as e:
                print(f"Error loading model details: {e}")
                return None, None, None, None, None, None, None
        
        # Connect models table
        models_table.select(
            load_model_details,
            inputs=[models_table.select_index, models_table],
            outputs=[
                model_id, model_name, model_type, model_status,
                model_description, model_capabilities, model_details
            ]
        )
        
        # Function to download model
        def download_model(model_id_value):
            try:
                if not model_id_value:
                    return "No model selected", None
                
                # Check if already downloaded
                if self.model_service.is_model_available_locally(model_id_value):
                    return f"Model {model_id_value} is already downloaded", "Available Locally"
                
                # Download model
                model_path = self.model_service.download_model(model_id_value)
                
                if model_path:
                    return f"Downloaded model {model_id_value} to {model_path}", "Available Locally"
                else:
                    return f"Failed to download model {model_id_value}", "Not Downloaded"
            
            except Exception as e:
                print(f"Error downloading model: {e}")
                return f"Error: {str(e)}", "Not Downloaded"
        
        # Connect download button
        download_model_button.click(
            download_model,
            inputs=[model_id],
            outputs=[gr.Textbox(visible=False), model_status]
        )
        
        # Load available models on start
        table_data = search_models("", "text_to_music")
        
        return [
            search_text, task_dropdown, search_button, models_table,
            model_id, model_name, model_type, model_status,
            model_description, model_capabilities, model_details,
            download_model_button, set_default_button
        ]
    
    def launch(self):
        """
        Launch the Gradio interface.
        """
        # Create interface
        with gr.Blocks(title="Disco Musica", theme=self.theme) as self.interface:
            # Header
            gr.Markdown(
                """
                # 🎵 Disco Musica
                ## AI Music Generation Platform
                
                Generate music from text, audio, or other input modalities using state-of-the-art AI models.
                """
            )
            
            # Main tabs
            with gr.Tabs():
                with gr.TabItem("Text to Music"):
                    self.text_to_music_interface()
                
                with gr.TabItem("Audio to Music"):
                    self.audio_to_music_interface()
                
                with gr.TabItem("Library"):
                    self.library_interface()
                
                with gr.TabItem("Models"):
                    self.model_management_interface()
                
                with gr.TabItem("Settings"):
                    gr.Markdown("Settings coming soon...")
            
            # Footer
            gr.Markdown(
                """
                ### About Disco Musica
                
                Disco Musica is an open-source multimodal AI music generation platform. 
                It leverages state-of-the-art machine learning models to generate music from various input modalities.
                
                [GitHub Repository](https://github.com/yourusername/disco-musica) | [Documentation](https://github.com/yourusername/disco-musica/docs)
                """
            )
        
        # Launch the interface
        self.interface.launch(share=False)


# Create and launch the interface
def launch_gradio_interface():
    """
    Create and launch the Gradio interface.
    """
    interface = GradioInterface()
    interface.launch()