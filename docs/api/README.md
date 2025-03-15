# Disco Musica API Documentation

This document provides detailed API documentation for the Disco Musica application.

## Table of Contents

1. [Core API](core.md)
2. [Generation API](generation.md)
3. [Training API](training.md)
4. [Processing API](processing.md)
5. [UI API](ui.md)

## Core API

The core API provides fundamental functionality for the Disco Musica application.

### Model Service

```python
from disco_musica.services.model_service import get_model_service

# Initialize model service
model_service = get_model_service()

# Load a model
model = model_service.load_model(model_name="musicgen")

# Get available models
available_models = model_service.list_available_models()
```

### Generation Service

```python
from disco_musica.services.generation_service import get_generation_service

# Initialize generation service
generation_service = get_generation_service()

# Generate music from text
audio = generation_service.generate_from_text(
    prompt="Create an upbeat disco track",
    duration=30,
    temperature=0.7
)
```

### Output Service

```python
from disco_musica.services.output_service import get_output_service

# Initialize output service
output_service = get_output_service()

# Save generated audio
output_service.save_audio(
    audio=audio,
    filename="generated_track.wav",
    format="wav"
)
```

## Generation API

The generation API provides functions for different types of music generation.

### Text-to-Music Generation

```python
from disco_musica.modules.inference import generate_from_text

# Generate music from text
audio = generate_from_text(
    prompt="Create a funky disco bassline",
    duration=15,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)
```

### Audio-to-Music Generation

```python
from disco_musica.modules.inference import generate_from_audio

# Generate music from audio
audio = generate_from_audio(
    input_audio="path/to/input.wav",
    style="disco",
    strength=0.8,
    duration=30
)
```

### MIDI-to-Music Generation

```python
from disco_musica.modules.inference import generate_from_midi

# Generate music from MIDI
audio = generate_from_midi(
    midi_file="path/to/input.mid",
    instruments=["bass", "drums", "synth"],
    duration=30
)
```

## Training API

The training API provides functions for model fine-tuning and training.

### Dataset Preparation

```python
from disco_musica.modules.training import prepare_dataset

# Prepare dataset
dataset = prepare_dataset(
    data_dir="path/to/music/files",
    format="audio",
    split_ratio=[0.8, 0.1, 0.1],
    preprocessing_config={
        "sample_rate": 44100,
        "duration": 30
    }
)
```

### Model Fine-tuning

```python
from disco_musica.modules.training import fine_tune_model

# Fine-tune model
model = fine_tune_model(
    base_model="musicgen",
    dataset=dataset,
    epochs=10,
    batch_size=32,
    learning_rate=1e-5
)
```

## Processing API

The processing API provides functions for audio and MIDI processing.

### Audio Processing

```python
from disco_musica.modules.audio_processing import process_audio

# Process audio
processed_audio = process_audio(
    audio=audio,
    sample_rate=44100,
    normalize=True,
    remove_silence=True
)
```

### MIDI Processing

```python
from disco_musica.modules.midi_processing import process_midi

# Process MIDI
processed_midi = process_midi(
    midi_file="path/to/input.mid",
    quantize=True,
    transpose=0
)
```

## UI API

The UI API provides functions for the graphical user interface.

### Launch UI

```python
from disco_musica.modules.interfaces.gradio_ui import launch_ui

# Launch the UI
launch_ui(debug=False)
```

### Custom UI Components

```python
from disco_musica.modules.interfaces.ui_components import create_generation_tab

# Create a custom generation tab
generation_tab = create_generation_tab(
    title="Music Generation",
    description="Generate music from various inputs"
)
```

## Error Handling

All API functions include proper error handling and logging. Errors are raised with descriptive messages and appropriate error codes.

```python
try:
    audio = generate_from_text(prompt="Create music")
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Generation failed: {e}")
```

## Logging

The API uses Python's logging module for consistent logging across the application.

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Starting music generation")
logger.error("Generation failed", exc_info=True)
```

## Configuration

The API can be configured using environment variables or a configuration file.

```python
from disco_musica.utils.config import load_config

# Load configuration
config = load_config("config.yaml")

# Use configuration
model_name = config["model"]["name"]
sample_rate = config["audio"]["sample_rate"]
``` 