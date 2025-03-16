# Disco Musica

An open-source multimodal AI music generation and analysis platform that leverages cutting-edge advancements in machine learning, signal processing, and time series analytics.

## Overview

Disco Musica aims to revolutionize music creation and analysis by providing a comprehensive suite of tools for AI-powered music generation, performance analysis, and data-driven insights. The platform offers unified interfaces for music generation, analysis, and model training, fostering a collaborative and innovative ecosystem.

## Key Features

### Multimodal Generation
- **Text-to-Music**: Generate music from natural language descriptions
- **Audio-to-Music**: Transform and generate music from existing audio
- **MIDI-to-Music**: Create music from MIDI files
- **Image-to-Music**: Generate music inspired by visual content
- **Video-to-Music**: Create music based on video content

### Advanced Analytics
- **Time Series Analysis**: Comprehensive analysis of performance metrics and training history
- **Pattern Detection**: Identify seasonal, cyclic, and trend patterns in musical data
- **Similarity Analysis**: Compare and analyze relationships between different metrics
- **Performance Monitoring**: Track and analyze model performance over time

### Data Management
- **Universal Storage**: Efficient storage for time series data and object files
- **Vector Embeddings**: Advanced similarity search and retrieval
- **Resource Tracking**: Comprehensive tracking of all system resources
- **Version Control**: Full version history for all resources

### Technical Features
- **Hybrid Architecture**: Local resources for inference, cloud resources for training
- **Model Fine-tuning**: Customize models for specific genres or styles
- **Efficient Processing**: Optimized for various hardware platforms
- **Open Source**: Community-driven development and improvement

## System Components

### Core Services
1. **Storage Services**
   - Time Series Storage: Performance metrics and training history
   - Object Storage: File management and versioning
   - Vector Storage: Similarity-based search and retrieval

2. **Model Services**
   - Production Model: Music generation and processing
   - Mastering Model: Audio mastering and enhancement
   - Vector Embedding Model: Semantic similarity analysis
   - Time Series Model: Pattern detection and forecasting

3. **Interface Services**
   - Logic Pro Interface: DAW integration
   - Vector Search Interface: Similarity search operations
   - Time Series Interface: Analytics and pattern detection

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Audio processing libraries (Librosa, PyDub)
- MIDI processing libraries (Music21)
- Data analysis libraries (NumPy, Pandas, SciPy)
- FFmpeg (for audio processing)

### Installation

```bash
# Clone the repository
git clone https://github.com/yavuztopsever/disco-musica.git
cd disco-musica

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage Examples

### Music Generation

```python
from disco_musica.models.production import ProductionModel
from disco_musica.core.config import ProductionConfig

# Initialize model
config = ProductionConfig(model_name="musicgen", device="cuda")
model = ProductionModel(config)

# Generate music
output = await model.generate(
    prompt="Create an upbeat disco track with funky bass",
    duration_seconds=30
)
```

### Time Series Analysis

```python
from disco_musica.interfaces.time_series import TimeSeriesInterface
from disco_musica.data.storage import TimeSeriesStorage

# Initialize interface
storage = TimeSeriesStorage()
interface = TimeSeriesInterface(storage)

# Analyze metrics
analysis = await interface.analyze_metric(
    metric_id="training_loss",
    start_time="2024-01-01",
    end_time="2024-03-15"
)

# Detect patterns
patterns = await interface.detect_patterns(
    metric_id="model_performance",
    pattern_type="seasonal",
    window_size=24
)
```

### Vector Search

```python
from disco_musica.interfaces.vector import VectorSearchInterface
from disco_musica.models.vector import VectorEmbeddingModel

# Initialize interface
model = VectorEmbeddingModel()
interface = VectorSearchInterface(model)

# Search similar items
results = await interface.search(
    query="upbeat electronic music with strong bass",
    top_k=10
)
```

## Project Structure

```
disco-musica/
├── modules/
│   ├── core/              # Core functionality and base classes
│   ├── data/             # Data storage and management
│   │   ├── storage/      # Storage implementations
│   │   └── processing/   # Data processing utilities
│   ├── interfaces/       # Interface implementations
│   │   ├── logic_pro/    # Logic Pro integration
│   │   ├── time_series/  # Time series analytics
│   │   └── vector/       # Vector search
│   ├── models/          # Model implementations
│   │   ├── production/   # Music generation
│   │   ├── mastering/    # Audio mastering
│   │   ├── vector/       # Embeddings
│   │   └── time_series/  # Time series analysis
│   └── services/        # Service implementations
├── docs/                # Documentation
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
└── README.md          # Project overview
```

## Documentation

- [Developer Guide](docs/developer_guide.md): Comprehensive technical documentation
- [System Flows](docs/flows.md): Detailed system workflows
- [Research Notes](docs/research/): Technical research and methodology

## Contributing

We welcome contributions! Please see our [Developer Guide](docs/developer_guide.md) for detailed information about the system architecture and development guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Meta's MusicGen for text-to-music generation
- Spotify's Basic Pitch for audio-to-MIDI conversion
- Hugging Face for model hosting and distribution
- The open-source AI music generation community