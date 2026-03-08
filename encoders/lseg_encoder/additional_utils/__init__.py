"""
Additional utilities for LSeg encoder optimization.

This package contains performance optimization modules for LSeg feature extraction:
- batch_processor: Batch processing engine
- async_io: Asynchronous I/O system
- data_preloader: Data preloading system
- encoding_models: Optimized model wrappers
"""

from .batch_processor import (
    BatchConfig,
    BatchFeatureExtractor,
    AsyncBatchProcessor,
)
from .async_io import (
    AsyncIOScheduler,
    AsyncFeatureSaver,
    IOStats,
)
from .data_preloader import (
    DataPreloader,
    BatchDataLoader,
    CachedImageLoader,
    create_preloading_dataloader,
)
from .encoding_models import (
    MultiEvalModule,
    vectorized_grid_inference,
)

__all__ = [
    # Batch processing
    'BatchConfig',
    'BatchFeatureExtractor',
    'AsyncBatchProcessor',
    # Async I/O
    'AsyncIOScheduler',
    'AsyncFeatureSaver',
    'IOStats',
    # Data preloading
    'DataPreloader',
    'BatchDataLoader',
    'CachedImageLoader',
    'create_preloading_dataloader',
    # Model utilities
    'MultiEvalModule',
    'vectorized_grid_inference',
]
