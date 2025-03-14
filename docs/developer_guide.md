# Disco Musica Developer Guide

Welcome to the Disco Musica Developer Guide. This document provides information for developers who want to contribute to the Disco Musica project or extend its functionality.

## Table of Contents

1. [Development Environment](#development-environment)
2. [Project Structure](#project-structure)
3. [Architecture](#architecture)
4. [Modules](#modules)
5. [Development Workflow](#development-workflow)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Contribution Guidelines](#contribution-guidelines)
9. [Code Style](#code-style)
10. [Resources](#resources)

## Development Environment

### Prerequisites

- Python 3.8+
- PyTorch or TensorFlow
- Audio processing libraries (Librosa, PyDub)
- MIDI processing libraries (Music21)
- Git

### Setup

1. Fork the repository on GitHub.
2. Clone your forked repository:
   ```bash
   git clone https://github.com/yourusername/disco-musica.git
   cd disco-musica
   ```

3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Project Structure

The project is organized into the following directories:

```
disco-musica/
├── data/
│   ├── raw/                  # Raw, unprocessed data
│   ├── processed/            # Preprocessed data
│   ├── datasets/             # Prepared datasets for training and inference
├── models/
│   ├── pretrained/           # Pre-trained models from external sources
│   ├── finetuned/            # User-finetuned models
├── modules/
│   ├── data_ingestion.py     # Data ingestion module
│   ├── preprocessing.py      # Data preprocessing module
│   ├── model_selection.py    # Model selection module
│   ├── training.py           # Training module
│   ├── inference.py          # Inference module
│   ├── ui.py                 # User interface module
│   ├── output_management.py   # Output management module
│   ├── audio_processing.py   # Audio processing module
│   ├── midi_processing.py    # MIDI processing module
├── utils/
│   ├── google_colab_utils.py # Utilities for Google Colab integration
│   ├── logging_utils.py      # Logging utilities
│   ├── cloud_utils.py        # Cloud platform utilities
├── notebooks/
│   ├── training_colab.ipynb  # Google Colab training notebook
│   ├── inference_demo.ipynb  # Inference demo notebook
├── tests/
│   ├── test_data_ingestion.py# Tests for data ingestion module
│   ├── test_preprocessing.py # Tests for data preprocessing module
│   ├── ...                   # Other test files
├── docs/
│   ├── documentation.md      # Main documentation file
│   ├── user_guide.md         # User guide
│   ├── developer_guide.md    # Developer guide
│   ├── api_docs/             # API documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Installation script
├── LICENSE                   # License file
├── README.md                 # Project README
```

## Architecture

Disco Musica employs a modular architecture to promote flexibility and maintainability. The application is divided into several modules, each responsible for a specific functionality.

### System Architecture

#### Local Application (User's Machine)

- **User Interface (UI) Layer**: Responsible for user interaction, input handling, and output display.
- **Inference Engine Layer**: Executes music generation based on user input and selected models.
- **Data Preprocessing Layer**: Handles data ingestion, preprocessing, and feature extraction.
- **Output Management Layer**: Manages generated music outputs, including saving, organizing, tagging, and exporting.

#### Cloud-Based Training Environment

- **Training Platform**: Provides access to powerful GPU resources for accelerated model training.
- **Model Fine-tuning Service**: Executes model fine-tuning based on user-provided datasets and training parameters.

## Modules

### Data Ingestion Module (`modules/data_ingestion.py`)

The Data Ingestion Module handles the ingestion of various music data formats (audio, MIDI, image, video) and provides functionalities for importing data from local files, cloud storage, and online repositories.

#### Key Classes and Functions

- `DataIngestionModule`: Main class for data ingestion operations.
  - `ingest_audio`: Ingests an audio file.
  - `ingest_midi`: Ingests a MIDI file.
  - `ingest_image`: Ingests an image file.
  - `ingest_video`: Ingests a video file.
  - `ingest_from_cloud`: Ingests a file from cloud storage.
  - `ingest_from_online_repo`: Ingests a file from an online repository.

### Preprocessing Module (`modules/preprocessing.py`)

The Preprocessing Module performs audio and MIDI preprocessing, including format conversion, segmentation, feature extraction, and stem separation. It also offers tools for dataset creation, splitting, and augmentation.

#### Key Classes and Functions

- `AudioPreprocessor`: Class for audio preprocessing operations.
  - `convert_format`: Converts audio file format.
  - `segment_audio`: Segments audio file into smaller chunks.
  - `extract_features`: Extracts features from audio.
  - `normalize_audio`: Normalizes audio.
  - `trim_silence`: Trims silence from audio.

- `MIDIPreprocessor`: Class for MIDI preprocessing operations.
  - `standardize_midi`: Standardizes MIDI file format and encoding.
  - `tokenize_midi`: Converts MIDI data into a sequence of tokens.
  - `quantize_midi`: Quantizes note timings to align with a musical grid.

- `DatasetManager`: Class for dataset creation and management.
  - `create_dataset`: Creates a dataset from a list of data paths.
  - `split_dataset`: Splits a dataset into training, validation, and testing sets.

### Training Module (`modules/training.py`)

The Training Module manages model fine-tuning, including training parameter adjustment, training progress monitoring, and checkpoint management. It integrates with cloud-based training environments (Google Colab, AWS, Azure).

#### Key Classes and Functions

- `TrainingManager`: Class for managing the training of music generation models.
  - `setup_training`: Sets up the training environment.
  - `prepare_dataset`: Prepares a dataset for training.
  - `fine_tune`: Fine-tunes a pre-trained model.
  - `parameter_efficient_fine_tune`: Fine-tunes a pre-trained model using parameter-efficient methods (LoRA).
  - `export_to_colab`: Exports the training setup to Google Colab.
  - `import_from_colab`: Imports a fine-tuned model from Google Colab.
  - `monitor_training_progress`: Monitors the progress of a training job.
  - `load_checkpoint`: Loads a training checkpoint.

### Inference Module (`modules/inference.py`)

The Inference Module executes music generation based on user input and selected models. It is optimized for efficient inference on local hardware.

#### Key Classes and Functions

- `InferenceEngine`: Class for executing music generation using trained models.
  - `load_model`: Loads a model for inference.
  - `unload_model`: Unloads a model from memory.
  - `generate_from_text`: Generates music from a text prompt.
  - `generate_from_audio`: Generates music from an audio file.
  - `generate_from_midi`: Generates music from a MIDI file.
  - `generate_from_image`: Generates music from an image file.
  - `adjust_parameters_realtime`: Adjusts generation parameters in real-time.
  - `save_output`: Saves generation output to a file.

### User Interface Module (`modules/ui.py`)

The User Interface Module provides a unified interface for both inference and training functionalities. It handles user interaction, input handling, and output display.

#### Key Classes and Functions

- `DiscoMusicaUI`: Class for the Disco Musica user interface.
  - `setup_inference_interface`: Sets up the inference interface.
  - `setup_training_interface`: Sets up the training interface.
  - `setup_model_management_interface`: Sets up the model management interface.
  - `launch_interface`: Launches the Disco Musica interface.
  - `text_to_music`: Generates music from a text prompt.
  - `audio_to_music`: Generates music from an audio input.
  - `train_model`: Trains a model on a dataset.

### Audio Processing Module (`modules/audio_processing.py`)

The Audio Processing Module provides functionalities for audio analysis, manipulation, and synthesis. It leverages libraries like Librosa and Pydub for audio processing tasks.

#### Key Classes and Functions

- `AudioProcessor`: Class for audio processing operations.
  - `load_audio`: Loads an audio file.
  - `save_audio`: Saves audio data to a file.
  - `convert_format`: Converts audio file format.
  - `change_sample_rate`: Changes the sample rate of an audio file.
  - `compute_spectrum`: Computes the spectrum of audio data.
  - `extract_features`: Extracts features from audio data.
  - `synthesize_audio`: Synthesizes audio from a spectrum.
  - `apply_effects`: Applies audio effects to audio data.
  - `mix_audio`: Mixes two audio tracks.
  - `normalize_audio`: Normalizes audio data.
  - `trim_silence`: Trims silence from audio data.

## Development Workflow

### Creating a New Feature

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Implement your feature, following the code style and architecture guidelines.

3. Write tests for your feature.

4. Run the tests to ensure they pass:
   ```bash
   pytest
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Add your feature"
   ```

6. Push your branch to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a pull request on GitHub.

### Fixing a Bug

1. Create a new branch for your bug fix:
   ```bash
   git checkout -b fix/your-bug-fix
   ```

2. Implement your bug fix, following the code style and architecture guidelines.

3. Write tests for your bug fix.

4. Run the tests to ensure they pass:
   ```bash
   pytest
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Fix your bug"
   ```

6. Push your branch to GitHub:
   ```bash
   git push origin fix/your-bug-fix
   ```

7. Create a pull request on GitHub.

## Testing

Disco Musica uses pytest for testing. Tests are located in the `tests/` directory.

### Running Tests

Run all tests:
```bash
pytest
```

Run tests for a specific module:
```bash
pytest tests/test_data_ingestion.py
```

Run tests with verbose output:
```bash
pytest -v
```

Run tests with coverage report:
```bash
pytest --cov=modules
```

### Writing Tests

When writing tests, follow these guidelines:

1. Use descriptive test names that explain what the test is checking.
2. Use fixtures to set up test data and resources.
3. Test both normal and edge cases.
4. Keep tests simple and focused on a single functionality.
5. Use assertions to check the result of the test.

## Documentation

Disco Musica uses Markdown for documentation. Documentation is located in the `docs/` directory.

### Writing Documentation

When writing documentation, follow these guidelines:

1. Use clear and concise language.
2. Use headings and lists to organize information.
3. Include code examples where appropriate.
4. Keep the documentation up-to-date with the code.

### Generating API Documentation

Disco Musica uses sphinx for API documentation. To generate API documentation:

```bash
cd docs
sphinx-build -b html api_docs api_docs_build
```

The generated documentation will be available in the `docs/api_docs_build/` directory.

## Contribution Guidelines

When contributing to Disco Musica, please follow these guidelines:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Implement your feature or bug fix, following the code style and architecture guidelines.
4. Write tests for your feature or bug fix.
5. Run the tests to ensure they pass.
6. Commit your changes with clear and concise commit messages.
7. Push your branch to GitHub.
8. Create a pull request on GitHub.

### Pull Request Process

1. Ensure all tests pass.
2. Update the documentation if needed.
3. Update the README.md if needed.
4. Request a review from one of the maintainers.

## Code Style

Disco Musica follows the PEP 8 style guide for Python code. Additionally, we use the following tools to enforce code style:

- Black: Code formatter
- Flake8: Linter
- isort: Import sorter

To format your code, run:
```bash
black .
isort .
```

To check your code for style issues, run:
```bash
flake8
```

## Resources

- [Python Documentation](https://docs.python.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Music21 Documentation](https://web.mit.edu/music21/doc/index.html)
- [Gradio Documentation](https://gradio.app/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)