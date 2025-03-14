# Disco Musica Documentation

Welcome to the Disco Musica documentation. This document provides comprehensive information about the Disco Musica application, including its architecture, features, and usage.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Architecture](#architecture)
4. [Modules](#modules)
5. [Inference](#inference)
6. [Training](#training)
7. [User Interface](#user-interface)
8. [API Documentation](#api-documentation)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction

Disco Musica is an open-source multimodal AI music generation application that leverages cutting-edge advancements in machine learning and multimodal AI. It aims to democratize music creation by making it accessible to a diverse range of users, from novice enthusiasts to seasoned professionals.

### Goals

- **Democratize Music Creation**: Empower users of all skill levels to explore and create music using AI.
- **Enhance Creative Workflows**: Provide professional musicians with advanced tools for experimentation, composition, and production.
- **Foster Community Collaboration**: Cultivate an open-source platform that encourages contributions, knowledge sharing, and continuous improvement.
- **Multimodal Input**: Support a rich variety of input modalities, including text, audio, MIDI, images, videos, and potentially biosensors.
- **Unified Interface**: Offer a single, intuitive interface for both inference (music generation) and training (model fine-tuning).
- **Efficient Training and Inference**: Facilitate fine-tuning of existing models on user-provided datasets and optimize inference for various hardware platforms.
- **Hybrid Architecture**: Leverage local resources for optimized inference and cloud resources (Google Colab and others) for computationally intensive training.
- **Ethical and Responsible AI**: Address ethical considerations related to copyright, bias, fairness, and the cultural impact of AI music generation.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch or TensorFlow
- Audio processing libraries (Librosa, PyDub)
- MIDI processing libraries (Music21)

### Installation Steps

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

## Architecture

Disco Musica employs a hybrid architecture to optimize performance and resource utilization, leveraging local resources for inference and cloud resources for training.

### System Architecture

#### Local Application (User's Machine)

- **User Interface (UI) Layer**: Responsible for user interaction, input handling, and output display.
- **Inference Engine Layer**: Executes music generation based on user input and selected models.
- **Data Preprocessing Layer**: Handles data ingestion, preprocessing, and feature extraction.
- **Output Management Layer**: Manages generated music outputs, including saving, organizing, tagging, and exporting.

#### Cloud-Based Training Environment

- **Training Platform**: Provides access to powerful GPU resources for accelerated model training.
- **Model Fine-tuning Service**: Executes model fine-tuning based on user-provided datasets and training parameters.

### Data Flow

1. **Data Ingestion**: The user uploads or imports their music data (audio, MIDI, image, video) through the Data Ingestion Module.
2. **Data Preprocessing**: The Data Preprocessing Module processes the ingested data, performing format conversion, segmentation, feature extraction, and other necessary preprocessing steps.
3. **Model Selection**: The user selects a pre-trained model for fine-tuning or inference through the Model Selection Module.
4. **Training (Cloud)**: If fine-tuning is desired, the preprocessed data and selected model are transferred to a cloud-based training environment (Google Colab, AWS, Azure) through the Training Module.
5. **Inference (Local)**: The user selects a trained model and provides input (text, audio, MIDI, etc.) through the User Interface Module.
6. **Output Management**: The Output Management Module provides functionalities for the user to interact with the generated music, including playback, visualization, saving, organizing, tagging, and exporting.

## Modules

### Data Ingestion Module

The Data Ingestion Module handles the ingestion of various music data formats (audio, MIDI, image, video) and provides functionalities for importing data from local files, cloud storage, and online repositories.

### Preprocessing Module

The Preprocessing Module performs audio and MIDI preprocessing, including format conversion, segmentation, feature extraction, and stem separation. It also offers tools for dataset creation, splitting, and augmentation.

### Model Selection Module

The Model Selection Module allows users to select pre-trained models for fine-tuning or inference and provides access to a curated selection of models from platforms like Hugging Face Hub.

### Training Module

The Training Module manages model fine-tuning, including training parameter adjustment, training progress monitoring, and checkpoint management. It integrates with cloud-based training environments (Google Colab, AWS, Azure).

### Inference Module

The Inference Module executes music generation based on user input and selected models. It is optimized for efficient inference on local hardware.

### User Interface (UI) Module

The User Interface Module provides a unified interface for both inference and training functionalities and handles user interaction, input handling, and output display.

### Output Management Module

The Output Management Module manages generated music outputs, including saving, organizing, tagging, and exporting. It provides functionalities for audio visualization, MIDI display, and symbolic notation.

### Audio Processing Module

The Audio Processing Module provides functionalities for audio analysis, manipulation, and synthesis. It leverages libraries like Librosa and Pydub for audio processing tasks.

### MIDI Processing Module

The MIDI Processing Module provides functionalities for MIDI parsing, manipulation, and generation. It leverages libraries like Music21 for MIDI processing tasks.

## Inference

Inference in Disco Musica refers to the process of generating music based on user input and selected models. The application supports various input modalities, including text, audio, MIDI, images, and videos.

### Text-to-Music

Generate music from natural language descriptions specifying genre, mood, instrumentation, tempo, key, and other musical attributes.

### Audio-to-Music

Generate music based on existing audio input, enabling:
- **Style Transfer**: Apply the stylistic characteristics of one piece of music to another.
- **Music Continuation**: Extend existing musical ideas or phrases.
- **Melody Conditioning**: Generate new musical arrangements based on provided melodies.
- **Instrumental Part Generation**: Generate specific instrumental parts that complement uploaded audio tracks.

### MIDI-to-Music

Generate music from MIDI files, supporting:
- **Style Transfer**: Apply stylistic characteristics of one MIDI file to another.
- **Harmonization**: Generate harmonies for existing melodies.
- **MIDI Continuation**: Extend or modify existing MIDI sequences.

### Image/Video-to-Music

Generate music inspired by the content or style of images and videos, exploring synesthetic relationships between visual and auditory stimuli.

## Training

Training in Disco Musica refers to the process of fine-tuning pre-trained models on user-provided datasets. The application integrates with cloud-based training environments to provide access to powerful GPU resources for accelerated model training.

### Dataset Creation

- **Data Ingestion**: Import music data from local files, cloud storage, or online repositories.
- **Data Preprocessing**: Process the ingested data, performing format conversion, segmentation, feature extraction, and other necessary preprocessing steps.
- **Dataset Management**: Split the dataset into training, validation, and testing sets, manage metadata, and implement data augmentation techniques.

### Model Fine-tuning

- **Pre-trained Model Selection**: Select a pre-trained model to fine-tune.
- **Training Parameter Adjustment**: Adjust key training parameters, including learning rate, batch size, number of epochs, optimizer selection, and loss function selection.
- **Parameter-Efficient Fine-tuning**: Implement LoRA or other parameter-efficient fine-tuning techniques to reduce computational cost and memory footprint.

### Cloud Integration

- **Google Colab Integration**: Transfer training data and model weights between the local machine and Google Colab, automate the process of setting up and running training jobs, and provide clear instructions and templates for using Google Colab for training.
- **Other Cloud Platforms**: Support other cloud platforms for training, including AWS and Azure.

## User Interface

The User Interface in Disco Musica provides a unified interface for both inference and training functionalities. It is designed to be intuitive and accessible to users of all skill levels.

### Inference Interface

- **Text-to-Music Interface**: Provide a text input field for entering natural language descriptions, controls for adjusting generation parameters, and a playback interface for the generated music.
- **Audio-to-Music Interface**: Allow users to upload audio files, select generation options, and provide a playback interface for the generated music.
- **MIDI-to-Music Interface**: Allow users to upload MIDI files, select generation options, and provide a playback interface for the generated music.
- **Image/Video-to-Music Interface**: Allow users to upload images or videos, select generation options, and provide a playback interface for the generated music.

### Training Interface

- **Dataset Creation Interface**: Provide tools for importing, preprocessing, and managing datasets.
- **Model Selection Interface**: Allow users to select pre-trained models for fine-tuning.
- **Training Parameter Interface**: Provide controls for adjusting training parameters.
- **Training Progress Interface**: Display real-time training progress, including loss function values, validation metrics, learning rate, and other relevant training information.

## API Documentation

Detailed API documentation for each module can be found in the [API Documentation](api_docs/) directory.

## Contributing

We welcome contributions from the community! Please see our [Contribution Guidelines](contributing.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.