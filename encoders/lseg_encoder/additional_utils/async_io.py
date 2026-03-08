"""
Asynchronous I/O System for LSeg Feature Extraction (Phase 2.2 optimization)

This module provides async file saving capabilities to prevent I/O operations
from blocking the main processing thread. This is particularly important when
saving large feature maps to disk.

Key optimizations:
- Background thread pool for file operations
- Non-blocking saves for feature tensors
- Progress tracking without blocking
- Error handling and retry logic
"""

import os
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field

import torch
import numpy as np
from PIL import Image


@dataclass
class SaveTask:
    """A single save task"""
    data: Any
    path: Path
    save_fn: Optional[Callable] = None
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class IOStats:
    """Statistics for I/O operations"""
    successful_saves: int = 0
    failed_saves: int = 0
    pending_count: int = 0
    total_bytes_saved: int = 0


class AsyncIOScheduler:
    """Asynchronous I/O scheduler for non-blocking file operations

    This class manages a pool of worker threads that handle file I/O
    operations in the background, preventing the main processing thread
    from being blocked by slow disk writes.

    Usage:
        scheduler = AsyncIOScheduler(max_workers=4)

        # Submit save operations (non-blocking)
        scheduler.submit_save(feature_tensor, output_path)

        # ... continue processing ...

        # Wait for all saves to complete
        scheduler.wait_completion()

        # Get statistics
        stats = scheduler.get_stats()
    """

    def __init__(
        self,
        max_workers: int = 4,
        queue_size: int = 32,
        enable_compression: bool = False
    ):
        """Initialize the async I/O scheduler

        Args:
            max_workers: Number of worker threads for file operations
            queue_size: Maximum size of the pending task queue
            enable_compression: Whether to compress saved tensors
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.enable_compression = enable_compression

        # Thread-safe counters
        self._lock = threading.Lock()
        self._stats = IOStats()
        self._error_log = []

        # Track pending tasks
        self._pending_tasks: Dict[str, SaveTask] = {}
        self._task_counter = 0

    def submit_save(
        self,
        data: Any,
        path: Path,
        save_fn: Optional[Callable] = None,
        task_id: Optional[str] = None
    ) -> str:
        """Submit a save task to the background queue

        Args:
            data: Data to save (torch.Tensor, numpy array, etc.)
            path: Output file path
            save_fn: Optional custom save function (default: auto-detect)
            task_id: Optional task identifier for tracking

        Returns:
            Task ID for tracking
        """
        path_obj = Path(path)

        if task_id is None:
            with self._lock:
                self._task_counter += 1
                task_id = f"task_{self._task_counter}"

        self._pending_tasks[task_id] = SaveTask(
            data=data,
            path=path_obj,
            save_fn=save_fn,
        )

        with self._lock:
            self._stats.pending_count += 1

        def save_task():
            task = self._pending_tasks.get(task_id)
            if task is None:
                return

            try:
                # Ensure parent directory exists
                task.path.parent.mkdir(parents=True, exist_ok=True)

                # Use custom save function or auto-detect
                if task.save_fn is not None:
                    task.save_fn(task.data, task.path)
                elif isinstance(task.data, torch.Tensor):
                    self._save_tensor(task.data, task.path)
                elif isinstance(task.data, np.ndarray):
                    self._save_numpy(task.data, task.path)
                elif isinstance(task.data, Image.Image):
                    task.data.save(task.path)
                else:
                    raise TypeError(f"Unsupported data type: {type(task.data)}")

                # Update stats
                with self._lock:
                    self._stats.successful_saves += 1
                    self._stats.pending_count -= 1
                    self._stats.total_bytes_saved += os.path.getsize(task.path)

                self._pending_tasks.pop(task_id, None)

            except Exception as e:
                # Retry before counting this task as failed.
                task = self._pending_tasks.get(task_id)
                if task is not None and task.retry_count < task.max_retries:
                    task.retry_count += 1
                    self.executor.submit(save_task)
                    return

                with self._lock:
                    self._stats.failed_saves += 1
                    self._stats.pending_count -= 1
                    self._error_log.append({
                        'task_id': task_id,
                        'path': str(path_obj),
                        'error': str(e),
                    })

                self._pending_tasks.pop(task_id, None)

        self.executor.submit(save_task)
        return task_id

    def _save_tensor(self, tensor: torch.Tensor, path: Path):
        """Save a torch.Tensor with appropriate format

        Args:
            tensor: Tensor to save
            path: Output path (determines format based on extension)
        """
        suffix = path.suffix.lower()

        if suffix == '.pt':
            # PyTorch format
            tensor_half = tensor.half() if tensor.dtype == torch.float32 else tensor
            torch.save(tensor_half, path)
        elif suffix in ('.png', '.jpg', '.jpeg'):
            # Image format
            if tensor.dim() == 3:  # [C, H, W]
                tensor = tensor.permute(1, 2, 0)  # [H, W, C]

            img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img_array).save(path)
        elif suffix == '.npy':
            # NumPy format
            np.save(path, tensor.cpu().numpy())
        elif suffix == '.npz':
            # Compressed NumPy format - 保存为 features 键（与加载逻辑匹配）
            np.savez_compressed(path, features=tensor.cpu().numpy())
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _save_numpy(self, array: np.ndarray, path: Path):
        """Save a numpy array with appropriate format

        使用 'features' 键保存 npz 文件，与加载逻辑匹配
        """
        suffix = path.suffix.lower()

        if suffix == '.npy':
            np.save(path, array)
        elif suffix == '.npz':
            np.savez_compressed(path, features=array)
        elif suffix == '.pt':
            torch.save(torch.from_numpy(array), path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def save_batch_features(
        self,
        features: list,
        paths: list,
        outdir: Path,
        format: str = 'pt'
    ):
        """Save a batch of feature tensors

        Args:
            features: List of feature tensors
            paths: List of original image paths (for naming)
            outdir: Output directory
            format: Output format ('pt', 'npy', 'npz')
        """
        suffix = f'.{format}' if not format.startswith('.') else format

        for feat, path in zip(features, paths):
            outname = Path(path).stem
            output_path = outdir / f"{outname}_fmap_CxHxW{suffix}"
            self.submit_save(feat, output_path)

    def save_batch_images(
        self,
        images: list,
        paths: list,
        outdir: Path,
        suffix: str = '_vis.png'
    ):
        """Save a batch of visualization images

        Args:
            images: List of image tensors [H, W, C] or [C, H, W]
            paths: List of original image paths
            outdir: Output directory
            suffix: Suffix to add to output filename
        """
        for img, path in zip(images, paths):
            outname = Path(path).stem
            output_path = outdir / f"{outname}{suffix}"
            self.submit_save(img, output_path)

    def wait_completion(self, timeout: Optional[float] = None):
        """Wait for all pending tasks to complete

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            True if all tasks completed, False if timeout
        """
        import time

        start_time = time.time()

        while self._stats.pending_count > 0:
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)  # Small sleep to avoid busy-waiting

        return True

    def get_stats(self) -> IOStats:
        """Get current I/O statistics

        Returns:
            IOStats object with current statistics
        """
        with self._lock:
            return IOStats(
                successful_saves=self._stats.successful_saves,
                failed_saves=self._stats.failed_saves,
                pending_count=self._stats.pending_count,
                total_bytes_saved=self._stats.total_bytes_saved,
            )

    def get_errors(self) -> list:
        """Get list of errors that occurred

        Returns:
            List of error dictionaries
        """
        with self._lock:
            return self._error_log.copy()

    def shutdown(self, wait: bool = True):
        """Shutdown the executor and cleanup resources

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        self.executor.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)


class AsyncFeatureSaver:
    """Specialized async saver for feature extraction pipeline

    Handles all the typical save operations for LSeg feature extraction.

    Usage:
        saver = AsyncFeatureSaver(output_dir)

        # Save features (non-blocking)
        saver.save_features(feature_tensor, image_name)

        # Save visualization (non-blocking)
        saver.save_visualization(vis_tensor, image_name)

        # Save prediction mask (non-blocking)
        saver.save_mask(mask_tensor, image_name)

        # Wait for all saves
        saver.wait_completion()
    """

    def __init__(
        self,
        output_dir: Path,
        max_workers: int = 4,
        save_format: str = 'pt'
    ):
        """Initialize the feature saver

        Args:
            output_dir: Root output directory
            max_workers: Number of I/O worker threads
            save_format: Default format for feature saves
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scheduler = AsyncIOScheduler(max_workers=max_workers)
        self.save_format = save_format

    def save_features(
        self,
        features: torch.Tensor,
        image_name: str,
        normalize: bool = False
    ) -> str:
        """Save feature tensor

        Args:
            features: Feature tensor [D, H, W] or [1, D, H, W]
            image_name: Base name for output file
            normalize: Whether to normalize before saving

        Returns:
            Task ID
        """
        if features.dim() == 4:
            features = features.squeeze(0)

        output_path = self.output_dir / f"{image_name}_fmap_CxHxW.{self.save_format}"

        if normalize:
            features = torch.nn.functional.normalize(features, dim=0)

        return self.scheduler.submit_save(features, output_path)

    def save_unnormalized_features(
        self,
        features: torch.Tensor,
        image_name: str
    ) -> str:
        """Save unnormalized feature tensor (original format)

        Args:
            features: Feature tensor [D, H, W]
            image_name: Base name for output file

        Returns:
            Task ID
        """
        if features.dim() == 4:
            features = features.squeeze(0)

        output_path = self.output_dir / f"{image_name}_fmap_unnormalized.{self.save_format}"

        # Save as float16 for efficiency
        return self.scheduler.submit_save(features.half(), output_path)

    def save_visualization(
        self,
        vis_tensor: torch.Tensor,
        image_name: str,
        suffix: str = '_feature_vis.png'
    ) -> str:
        """Save visualization image

        Args:
            vis_tensor: Visualization tensor [H, W, 3] in [0, 1]
            image_name: Base name for output file
            suffix: Suffix to add to filename

        Returns:
            Task ID
        """
        output_path = self.output_dir / f"{image_name}{suffix}"

        def save_vis(data, path):
            img_array = (data.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img_array).save(path)

        return self.scheduler.submit_save(vis_tensor, output_path, save_vis)

    def save_mask(
        self,
        mask: torch.Tensor,
        image_name: str,
        palette: list = None
    ) -> str:
        """Save prediction mask

        Args:
            mask: Mask tensor [H, W] with class indices
            image_name: Base name for output file
            palette: Optional color palette

        Returns:
            Task ID
        """
        output_path = self.output_dir / f"{image_name}.png"

        def save_mask_with_palette(data, path):
            mask_array = data.cpu().numpy().astype(np.uint8)
            img = Image.fromarray(mask_array)

            if palette:
                img.putpalette(palette)

            img.save(path)

        return self.scheduler.submit_save(mask, output_path, save_mask_with_palette)

    def save_pca_dict(
        self,
        pca_dict: dict,
        filename: str = "pca_dict.pt"
    ) -> str:
        """Save PCA parameters

        Args:
            pca_dict: Dictionary with PCA parameters
            filename: Output filename

        Returns:
            Task ID
        """
        output_path = self.output_dir / filename
        return self.scheduler.submit_save(pca_dict, output_path)

    def wait_completion(self, timeout: Optional[float] = None):
        """Wait for all saves to complete"""
        return self.scheduler.wait_completion(timeout)

    def get_stats(self) -> IOStats:
        """Get I/O statistics"""
        return self.scheduler.get_stats()

    def shutdown(self):
        """Shutdown the scheduler"""
        self.scheduler.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
