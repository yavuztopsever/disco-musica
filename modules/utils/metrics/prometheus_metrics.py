"""Prometheus metrics for monitoring."""
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
REQUESTS_TOTAL = Counter(
    'disco_musica_requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)

RESPONSE_TIME = Histogram(
    'disco_musica_response_time_seconds',
    'Response time in seconds',
    ['endpoint'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

MODEL_INFERENCE_TIME = Histogram(
    'disco_musica_model_inference_time_seconds',
    'Model inference time in seconds',
    ['model_type', 'model_id'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)

GENERATION_SIZES = Histogram(
    'disco_musica_generation_sizes_bytes',
    'Size of generated content in bytes',
    ['content_type'],
    buckets=(1024, 10*1024, 100*1024, 1024*1024, 10*1024*1024)
)

MODEL_TRAINING_TIME = Histogram(
    'disco_musica_model_training_time_seconds',
    'Model training time in seconds',
    ['model_type', 'model_id'],
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800)
)

TRAINING_LOSS = Histogram(
    'disco_musica_training_loss',
    'Training loss values',
    ['model_type', 'model_id'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

VALIDATION_LOSS = Histogram(
    'disco_musica_validation_loss',
    'Validation loss values',
    ['model_type', 'model_id'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Start metrics server
def start_metrics_server(port=8001):
    """Start Prometheus metrics server.
    
    Args:
        port: Server port.
    """
    start_http_server(port) 