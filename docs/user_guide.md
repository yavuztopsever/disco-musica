# Disco Musica User Guide

Welcome to the Disco Musica User Guide. This document provides instructions on how to use the Disco Musica application, including music generation (inference) and model fine-tuning (training).

## Table of Contents

1. [Getting Started](#getting-started)
2. [Music Generation (Inference)](#music-generation-inference)
3. [Model Fine-tuning (Training)](#model-fine-tuning-training)
4. [Managing Generated Music](#managing-generated-music)
5. [Frequently Asked Questions](#frequently-asked-questions)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/disco-musica.git
   cd disco-musica
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python main.py
   ```

### First-time Setup

When you first run the application, you'll be presented with the Disco Musica interface. The interface is divided into tabs for different functionalities:

- **Text-to-Music**: Generate music from text descriptions.
- **Audio-to-Music**: Generate music from audio input.
- **MIDI-to-Music**: Generate music from MIDI files.
- **Image-to-Music**: Generate music from images.
- **Model Training**: Fine-tune models on your own data.
- **Model Management**: Manage your models.

## Music Generation (Inference)

### Text-to-Music

1. Navigate to the **Text-to-Music** tab.
2. Enter a text description of the music you want to generate in the **Text Prompt** field.
   - Be as specific as possible in your description, mentioning genre, mood, instrumentation, tempo, etc.
   - Example: "A jazz piece with piano and drums, medium tempo, melancholic mood."
3. Select a model from the **Model** dropdown.
4. Adjust the generation parameters:
   - **Temperature**: Controls the randomness of the generation. Higher values (e.g., 1.5) will produce more creative and unexpected results, while lower values (e.g., 0.5) will produce more deterministic and safe results.
   - **Top-p**: Controls the diversity of the generation. Lower values will make the generation more focused, while higher values will make it more diverse.
   - **Maximum Length**: Controls the length of the generated music in tokens.
5. Click the **Generate Music** button.
6. The generated music will be displayed in the right panel. You can play it, visualize the waveform, and download it.

### Audio-to-Music

1. Navigate to the **Audio-to-Music** tab.
2. Upload an audio file by clicking on the **Input Audio** field.
3. Select a model from the **Model** dropdown.
4. Choose a style transfer option from the **Style Transfer** dropdown, if desired.
5. Click the **Generate Music** button.
6. The generated music will be displayed in the right panel. You can play it, visualize the waveform, and download it.

### MIDI-to-Music

1. Navigate to the **MIDI-to-Music** tab.
2. Upload a MIDI file by clicking on the **Input MIDI** field.
3. Select a model from the **Model** dropdown.
4. Choose an instrument from the **Instrument** dropdown.
5. Click the **Generate Music** button.
6. The generated music will be displayed in the right panel. You can play it, visualize the MIDI, and download it.

### Image-to-Music

1. Navigate to the **Image-to-Music** tab.
2. Upload an image by clicking on the **Input Image** field.
3. Select a model from the **Model** dropdown.
4. Choose a mood from the **Mood** dropdown, or leave it as **Automatic** to let the model determine the mood from the image.
5. Click the **Generate Music** button.
6. The generated music will be displayed in the right panel. You can play it, visualize the waveform, and download it.

## Model Fine-tuning (Training)

### Preparing Your Dataset

1. Collect your music data (audio, MIDI, etc.) and organize it in a directory.
2. Make sure the data is in a compatible format (e.g., WAV, MP3, FLAC, MIDI).
3. Consider preprocessing your data to ensure consistency (e.g., same sample rate, bit depth, etc.).

### Training a Model

1. Navigate to the **Model Training** tab.
2. Enter a name for your dataset in the **Dataset Name** field.
3. Upload your dataset files by clicking on the **Dataset Files** field.
4. Select a base model from the **Base Model** dropdown.
5. Adjust the training parameters:
   - **Epochs**: Number of training epochs.
   - **Batch Size**: Training batch size.
   - **Learning Rate**: Learning rate for the optimizer.
6. Choose a training type:
   - **Full Fine-tuning**: Update all model parameters.
   - **Parameter-efficient Fine-tuning (LoRA)**: Use LoRA to reduce computational cost and memory footprint.
7. Choose a training platform:
   - **Local**: Train on your local machine.
   - **Google Colab**: Train on Google Colab.
   - **AWS**: Train on Amazon Web Services.
   - **Azure**: Train on Microsoft Azure.
8. Click the **Start Training** button.
9. The training progress will be displayed in the right panel. You can monitor the training status, progress, and logs.

### Using Your Fine-tuned Model

1. Navigate to the **Model Management** tab.
2. Find your model in the list of available models.
3. Select it to view details and use it for inference.
4. You can now use your fine-tuned model in any of the music generation tabs.

## Managing Generated Music

### Saving Generated Music

1. After generating music, you can save it by clicking the **Download** button.
2. Choose a location and filename for the saved music.
3. The music will be saved in the selected format (e.g., WAV, MP3, MIDI).

### Organizing Your Music

1. Disco Musica provides tools for organizing your generated music.
2. You can create playlists, add tags, and categorize your music.
3. This helps you keep track of your generated music and find it easily.

## Frequently Asked Questions

### What models are available?

Disco Musica comes with a selection of pre-trained models for music generation. These models are based on state-of-the-art architectures and have been trained on large datasets of music.

### Can I use my own models?

Yes, you can use your own models with Disco Musica. You can either fine-tune existing models or import compatible models from other sources.

### What file formats are supported?

Disco Musica supports a wide range of audio and MIDI formats, including:
- Audio: WAV, MP3, FLAC, OGG
- MIDI: Standard MIDI files (.mid, .midi)
- Images: JPEG, PNG
- Video: MP4, MOV

### How long does training take?

Training time depends on various factors, including the size of your dataset, the complexity of the model, and the hardware you're using. Training on a GPU is significantly faster than on a CPU. Training on cloud platforms like Google Colab can be faster than local training, depending on the available resources.

## Troubleshooting

### Installation Issues

If you encounter issues during installation, make sure you have the required dependencies installed. Some libraries, like Librosa, may require additional system packages.

### Runtime Errors

If you encounter runtime errors, check the error message for details. Common issues include:
- Memory limitations: Try reducing batch size or model size.
- GPU issues: Make sure your GPU drivers are up to date.
- File format issues: Make sure your input files are in a supported format.

### Performance Issues

If the application is running slowly, consider:
- Using a GPU for inference and training.
- Reducing the complexity of the models you're using.
- Optimizing your input data (e.g., reducing audio file size).

### Getting Help

If you need help, you can:
- Check the [documentation](documentation.md) for detailed information.
- Open an issue on the GitHub repository.
- Reach out to the community for support.