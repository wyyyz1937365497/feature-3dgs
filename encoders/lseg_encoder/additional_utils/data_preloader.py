"""
Data Preloading System for LSeg Feature Extraction (Phase 2.3 optimization)

This module provides data prefetching capabilities to overlap I/O operations
with GPU computation, significantly reducing the time spent waiting for data.

Key optimizations:
- Prefetch queue for overlapping I/O with compute
- GPU memory preloading with non-blocking transfers
- Smart caching for frequently accessed images
- Memory-efficient buffer management
"""

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class PrefetchConfig:
    """Configuration for data preloading"""
    prefetch_size: int = 4  # Number of images to prefetch
    device: str = 'cuda'  # Target device for preloaded data
    enable_gpu_cache: bool = True  # Cache preloaded tensors on GPU
    cache_size: int = 32  # Maximum number of cached images


class DataPreloader:
    """Data preloader with background thread and GPU caching

    This class maintains a queue of preloaded images, loading them in the
    background while the GPU processes the current batch.

    Usage:
        preloader = DataPreloader(
            image_paths,
            transform=your_transform,
            prefetch_size=4
        )
        preloader.start()

        # Get preloaded images (non-blocking)
        while True:
            result = preloader.get_next()
            if result is None:
                break
            idx, img_tensor = result
            # Process img_tensor...

        preloader.stop()
    """

    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None,
        prefetch_size: int = 4,
        device: str = 'cuda',
        enable_cache: bool = True
    ):
        """Initialize the data preloader

        Args:
            image_paths: List of paths to image files
            transform: Optional transform to apply to images
            prefetch_size: Number of images to prefetch ahead
            device: Target device ('cuda' or 'cpu')
            enable_cache: Whether to enable GPU-side caching
        """
        self.image_paths = image_paths
        self.transform = transform or (lambda x: x)
        self.prefetch_size = prefetch_size
        self.device = device
        self.enable_cache = enable_cache

        # Thread-safe queue for preloaded data
        self.queue = queue.Queue(maxsize=prefetch_size)

        # GPU cache for frequently accessed images
        self._cache: Dict[str, torch.Tensor] = {}
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

        # Thread control
        self.running = False
        self.loader_thread = None
        self.current_index = 0

        # Statistics
        self._load_times = []
        self._total_loaded = 0

    def start(self):
        """Start the background prefetch thread"""
        self.running = True
        self.current_index = 0
        self.loader_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.loader_thread.start()

    def stop(self):
        """Stop the prefetch thread"""
        self.running = False
        if self.loader_thread:
            self.loader_thread.join(timeout=2.0)

    def _prefetch_worker(self):
        """Background worker thread for prefetching images"""
        while self.running and self.current_index < len(self.image_paths):
            try:
                # Check if queue is full
                if self.queue.full():
                    time.sleep(0.01)
                    continue

                # Load next image
                img_path = self.image_paths[self.current_index]
                start_time = time.time()

                # Check cache first
                cached = self._get_from_cache(img_path)
                if cached is not None:
                    img_tensor = cached
                    self._cache_hits += 1
                else:
                    # Load image
                    img = self._load_image(img_path)
                    img_tensor = self._transform_and_transfer(img)

                    # Add to cache if enabled
                    if self.enable_cache:
                        self._add_to_cache(img_path, img_tensor)

                    self._cache_misses += 1

                load_time = time.time() - start_time
                self._load_times.append(load_time)
                self._total_loaded += 1

                # Put in queue (non-blocking)
                try:
                    self.queue.put((self.current_index, img_tensor), block=False)
                except queue.Full:
                    # Queue is full, discard this image
                    pass

                self.current_index += 1

            except Exception as e:
                print(f"Error prefetching {self.image_paths[self.current_index]}: {e}")
                self.current_index += 1

    def _load_image(self, path: str) -> Image.Image:
        """Load image from disk"""
        img = Image.open(path).convert('RGB')
        return img

    def _transform_and_transfer(self, img: Image.Image) -> torch.Tensor:
        """Apply transform and transfer to target device"""
        img_tensor = self.transform(img)

        if self.device.startswith('cuda'):
            # Non-blocking transfer to GPU
            img_tensor = img_tensor.to(self.device, non_blocking=True)

        return img_tensor

    def _get_from_cache(self, path: str) -> Optional[torch.Tensor]:
        """Get image from cache if available"""
        if not self.enable_cache:
            return None

        with self._cache_lock:
            return self._cache.get(path)

    def _add_to_cache(self, path: str, tensor: torch.Tensor):
        """Add image to cache"""
        if not self.enable_cache:
            return

        with self._cache_lock:
            # Simple cache size management
            if len(self._cache) >= 32:  # Max cache size
                # Remove oldest entry
                oldest = next(iter(self._cache))
                del self._cache[oldest]

            self._cache[path] = tensor.clone()  # Clone to avoid modifications

    def get_next(self, timeout: float = 5.0) -> Optional[Tuple[int, torch.Tensor]]:
        """Get next preloaded image

        Args:
            timeout: Maximum time to wait for data

        Returns:
            (index, image_tensor) or None if no more data
        """
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get preloading statistics

        Returns:
            Dictionary with statistics
        """
        avg_load_time = sum(self._load_times) / len(self._load_times) if self._load_times else 0
        cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0

        return {
            'total_loaded': self._total_loaded,
            'avg_load_time_ms': avg_load_time * 1000,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._cache),
        }


class SmartDataPreloader(DataPreloader):
    """Enhanced preloader with smart batch size adaptation

    This version adapts the prefetch size based on GPU memory usage
    and loading times to optimize throughput.
    """

    def __init__(self, *args, initial_batch_size: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = initial_batch_size
        self.adaptive_mode = True
        self.memory_usage_samples = []

    def get_next_batch(self, max_size: Optional[int] = None) -> Tuple[List[int], List[torch.Tensor]]:
        """Get a batch of preloaded images

        Args:
            max_size: Maximum batch size (defaults to current adaptive batch size)

        Returns:
            (indices, tensors) lists
        """
        batch_size = max_size or self.batch_size
        indices = []
        tensors = []

        for _ in range(batch_size):
            result = self.get_next(timeout=1.0)
            if result is None:
                break
            idx, tensor = result
            indices.append(idx)
            tensors.append(tensor)

        # Adaptive batch size adjustment
        if self.adaptive_mode and len(tensors) < batch_size:
            # Reduce batch size if we're running low on preloaded data
            self.batch_size = max(1, len(tensors))

        return indices, tensors


class BatchDataLoader:
    """Simplified batch data loader with integrated preloading

    This class combines dataloader functionality with preloading
    for a drop-in replacement in existing pipelines.

    Usage:
        loader = BatchDataLoader(image_paths, batch_size=4, transform=transform)
        loader.start()

        for batch_indices, batch_images in loader:
            # Process batch_images...
            pass

        loader.stop()
    """

    def __init__(
        self,
        image_paths: List[str],
        batch_size: int = 4,
        transform: Optional[Callable] = None,
        device: str = 'cuda',
        prefetch_batches: int = 2,
        shuffle: bool = False
    ):
        """Initialize the batch data loader

        Args:
            image_paths: List of image file paths
            batch_size: Number of images per batch
            transform: Optional image transform
            device: Target device
            prefetch_batches: Number of batches to prefetch
            shuffle: Whether to shuffle the data
        """
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.transform = transform
        self.device = device
        self.prefetch_batches = prefetch_batches
        self.shuffle = shuffle

        # Create index list
        self.indices = list(range(len(image_paths)))
        if shuffle:
            import random
            random.shuffle(self.indices)

        # Preloader
        self.preloader = None
        self.running = False

    def start(self):
        """Start the data loader"""
        self.running = True

        # Create ordered paths based on indices
        ordered_paths = [self.image_paths[i] for i in self.indices]

        self.preloader = DataPreloader(
            image_paths=ordered_paths,
            transform=self.transform or (lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1).float() / 127.5 - 1),
            prefetch_size=self.batch_size * self.prefetch_batches,
            device=self.device
        )
        self.preloader.start()

    def stop(self):
        """Stop the data loader"""
        self.running = False
        if self.preloader:
            self.preloader.stop()

    def __iter__(self):
        """Iterate over batches"""
        if not self.running:
            self.start()

        while True:
            # Collect a batch
            indices = []
            images = []

            for _ in range(self.batch_size):
                result = self.preloader.get_next(timeout=5.0)
                if result is None:
                    break
                idx, img = result
                indices.append(idx)
                images.append(img)

            if not images:
                break

            # Stack images into a batch
            batch = torch.stack(images, dim=0)
            yield indices, batch

    def __len__(self):
        """Number of batches"""
        return (len(self.image_paths) + self.batch_size - 1) // self.batch_size

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics"""
        if self.preloader:
            return self.preloader.get_stats()
        return {}


class CachedImageLoader:
    """Memory-efficient image loader with LRU cache

    Caches recently loaded images in CPU memory to avoid
    redundant disk reads for repeated access patterns.

    Usage:
        loader = CachedImageLoader(cache_size=128)
        img = loader.load(image_path)
    """

    def __init__(self, cache_size: int = 128):
        """Initialize the cached loader

        Args:
            cache_size: Maximum number of images to cache
        """
        self.cache_size = cache_size
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()

    def load(self, path: str, transform: Optional[Callable] = None) -> Image.Image:
        """Load image with caching

        Args:
            path: Path to image file
            transform: Optional transform to apply

        Returns:
            Loaded (and transformed) image
        """
        # Check cache
        with self._lock:
            if path in self._cache:
                # Update access order
                self._access_order.remove(path)
                self._access_order.append(path)
                return self._cache[path]

        # Load image
        img = Image.open(path).convert('RGB')

        # Apply transform if provided
        if transform:
            img = transform(img)

        # Add to cache
        with self._lock:
            if len(self._cache) >= self.cache_size:
                # Remove least recently used
                lru_path = self._access_order.pop(0)
                del self._cache[lru_path]

            self._cache[path] = img
            self._access_order.append(path)

        return img

    def clear(self):
        """Clear the cache"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()


def create_preloading_dataloader(
    image_paths: List[str],
    batch_size: int = 4,
    transform: Optional[Callable] = None,
    device: str = 'cuda',
    num_workers: int = 0
) -> BatchDataLoader:
    """Factory function to create a preloading dataloader

    This is a convenience function that creates a BatchDataLoader
    with sensible defaults for feature extraction.

    Args:
        image_paths: List of image file paths
        batch_size: Batch size for processing
        transform: Optional image transform
        device: Target device for tensors
        num_workers: (Unused, kept for API compatibility)

    Returns:
        Configured BatchDataLoader instance
    """
    return BatchDataLoader(
        image_paths=image_paths,
        batch_size=batch_size,
        transform=transform,
        device=device,
        prefetch_batches=2,
        shuffle=False
    )
