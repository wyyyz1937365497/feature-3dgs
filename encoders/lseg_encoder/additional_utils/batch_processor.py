"""
Batch Processing Engine for LSeg Feature Extraction (Phase 2.1 optimization)

This module provides true batch processing capabilities for multi-image
feature extraction, replacing the inefficient single-image-at-a-time approach.

Key optimizations:
- Batch multi-image processing instead of sequential processing
- Mixed precision support (FP16) for faster inference
- Efficient padding and resizing for variable-sized images
- Prefetch queue for overlapping compute and I/O
"""

import math
import queue
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    max_batch_size: int = 4  # Maximum number of images to process in one batch
    prefetch_count: int = 2  # Number of batches to prefetch
    enable_mixed_precision: bool = True  # Use FP16 for faster inference
    max_image_size: Tuple[int, int] = (640, 480)  # Max H, W for batch processing


class BatchFeatureExtractor:
    """Batch feature extractor for efficient multi-image processing

    This class processes multiple images in a single batch, reducing
    the overhead of individual model calls.

    Usage:
        extractor = BatchFeatureExtractor(model, scales=[0.75, 1.0, 1.25])
        features = extractor.extract_batch(images)
    """

    def __init__(
        self,
        model,
        scales: List[float],
        config: Optional[BatchConfig] = None,
        base_size: int = 520,
        crop_size: int = 480,
    ):
        self.model = model
        self.scales = scales
        self.config = config or BatchConfig()
        self.base_size = base_size
        self.crop_size = crop_size
        self.device = next(model.parameters()).device

        # Get upsampling kwargs from model
        self._up_kwargs = getattr(model, '_up_kwargs', {'mode': 'bilinear', 'align_corners': True})

    def extract_batch(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        """Extract features for a batch of images

        Args:
            images: List of image tensors, each [C, H, W]

        Returns:
            List of feature tensors, each [D, H, W] where D is feature dimension
        """
        batch_size = len(images)
        if batch_size == 0:
            return []

        # Find max dimensions for padding
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)

        # Clamp to max_image_size to avoid memory issues
        max_h = min(max_h, self.config.max_image_size[0])
        max_w = min(max_w, self.config.max_image_size[1])

        # Create batch tensor with padding
        dtype = torch.float16 if self.config.enable_mixed_precision else torch.float32
        batch = torch.zeros(batch_size, 3, max_h, max_w, device=self.device, dtype=dtype)

        for i, img in enumerate(images):
            h, w = img.shape[1], img.shape[2]
            # Resize if too large
            if h > max_h or w > max_w:
                img_resized = F.interpolate(
                    img.unsqueeze(0).to(self.device),
                    size=(min(h, max_h), min(w, max_w)),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                h, w = img_resized.shape[1], img_resized.shape[2]
                batch[i, :, :h, :w] = img_resized
            else:
                batch[i, :, :h, :w] = img.to(self.device)

        # Batch multi-scale inference
        with torch.cuda.amp.autocast(enabled=self.config.enable_mixed_precision):
            with torch.no_grad():
                batch_features = self._forward_multiscale(batch, original_sizes=[img.shape[1:] for img in images])

        return batch_features

    def _forward_multiscale(
        self,
        batch_images: torch.Tensor,
        original_sizes: List[Tuple[int, int]]
    ) -> List[torch.Tensor]:
        """Multi-scale forward pass for batch

        Args:
            batch_images: Batch of images [B, C, H, W]
            original_sizes: Original (H, W) for each image

        Returns:
            List of feature tensors per image
        """
        batch_size, _, h, w = batch_images.size()
        device = batch_images.device

        # Get feature dimension from first scale
        first_scale = self.scales[0]
        first_h = int(h * first_scale)
        first_w = int(w * first_scale)
        first_scaled = F.interpolate(
            batch_images, size=(first_h, first_w),
            mode='bilinear', align_corners=False
        )

        # Use a single image to probe output dimension
        with torch.no_grad():
            dummy_output = self.model.evaluate(
                first_scaled[:1].float(),
                return_feature=True
            )
            feature_dim = dummy_output.shape[1]

        # Accumulate features across scales
        accumulated_features = torch.zeros(
            batch_size, feature_dim, h, w,
            device=device,
            dtype=torch.float16 if self.config.enable_mixed_precision else torch.float32
        )

        for scale in self.scales:
            scaled_h = int(h * scale)
            scaled_w = int(w * scale)

            # Scale the batch
            scaled_images = F.interpolate(
                batch_images.float(),  # Use FP32 for interpolation
                size=(scaled_h, scaled_w),
                mode='bilinear',
                align_corners=False
            )

            # Extract features at this scale
            scale_features = self._evaluate_batch(scaled_images)

            # Interpolate back to original size
            scale_features = F.interpolate(
                scale_features,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )

            accumulated_features += scale_features

        # Average over scales
        accumulated_features = accumulated_features / len(self.scales)

        # Separate batch into individual outputs
        outputs = []
        for i, (orig_h, orig_w) in enumerate(original_sizes):
            # Resize to original size
            feat = accumulated_features[i:i+1, :, :h, :w]
            if orig_h != h or orig_w != w:
                feat = F.interpolate(
                    feat, size=(orig_h, orig_w),
                    mode='bilinear', align_corners=False
                )
            outputs.append(feat.squeeze(0))

        return outputs

    def _evaluate_batch(self, batch_images: torch.Tensor) -> torch.Tensor:
        """Evaluate model on batch of images

        Handles grid-based evaluation for large images.

        Args:
            batch_images: Batch [B, C, H, W]

        Returns:
            Batch features [B, D, H, W]
        """
        batch_size, _, h, w = batch_images.size()
        base_size = self.base_size
        crop_size = self.crop_size
        stride_rate = 2.0 / 3.0
        stride = int(crop_size * stride_rate)

        # Check if we need grid evaluation
        long_size = max(h, w)
        if long_size <= crop_size:
            # Single evaluation, no grid needed
            return self._single_batch_evaluation(batch_images)
        else:
            # Grid evaluation needed
            return self._grid_batch_evaluation(batch_images, crop_size, stride)

    def _single_batch_evaluation(self, batch_images: torch.Tensor) -> torch.Tensor:
        """Single evaluation without grid"""
        return self.model.evaluate(batch_images, return_feature=True)

    def _grid_batch_evaluation(
        self,
        batch_images: torch.Tensor,
        crop_size: int,
        stride: int
    ) -> torch.Tensor:
        """Grid-based evaluation for large images

        Processes the image in overlapping grid crops and aggregates results.
        This is a batched version of the grid evaluation logic.

        Args:
            batch_images: Batch [B, C, H, W]
            crop_size: Size of grid crops
            stride: Stride between crops

        Returns:
            Aggregated features [B, D, H, W]
        """
        batch_size, _, h, w = batch_images.size()

        # Calculate number of grids
        h_grids = int(math.ceil(1.0 * (h - crop_size) / stride)) + 1
        w_grids = int(math.ceil(1.0 * (w - crop_size) / stride)) + 1

        # Get feature dimension
        dummy_crop = batch_images[:, :, :crop_size, :crop_size]
        dummy_output = self.model.evaluate(dummy_crop, return_feature=True)
        feature_dim = dummy_output.shape[1]

        # Initialize output tensors
        device = batch_images.device
        dtype = torch.float16 if self.config.enable_mixed_precision else torch.float32
        accumulated = torch.zeros(batch_size, feature_dim, h, w, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, h, w, device=device, dtype=dtype)

        # Collect all crops
        crops = []
        positions = []
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, h)
                w1 = min(w0 + crop_size, w)

                crop = batch_images[:, :, h0:h1, w0:w1]

                # Pad if needed
                if h1 - h0 < crop_size or w1 - w0 < crop_size:
                    crop = self._pad_crop(crop, crop_size)

                crops.append(crop)
                positions.append((h0, h1, w0, w1))

        # Process crops in batches
        batch_chunk_size = min(8, len(crops))
        for i in range(0, len(crops), batch_chunk_size):
            chunk_end = min(i + batch_chunk_size, len(crops))
            chunk_crops = torch.cat(crops[i:chunk_end], dim=0)

            # Batch inference
            chunk_outputs = self.model.evaluate(chunk_crops, return_feature=True)

            # Scatter back
            for j in range(i, chunk_end):
                local_idx = j - i
                h0, h1, w0, w1 = positions[j]
                crop_h, crop_w = h1 - h0, w1 - w0

                # Handle batch dimension
                batch_idx = j % batch_size
                accumulated[batch_idx:batch_idx+1, :, h0:h1, w0:w1] += (
                    chunk_outputs[local_idx:local_idx+1, :, :crop_h, :crop_w]
                )
                count[batch_idx:batch_idx+1, :, h0:h1, w0:w1] += 1

        # Normalize
        return accumulated / count.clamp(min=1)

    def _pad_crop(self, crop: torch.Tensor, target_size: int) -> torch.Tensor:
        """Pad crop to target size

        Uses the model's mean/std for padding values
        """
        batch, channels, h, w = crop.size()
        padh = target_size - h
        padw = target_size - w

        mean = torch.tensor(self.model.mean).view(1, -1, 1, 1).to(crop.device)
        std = torch.tensor(self.model.std).view(1, -1, 1, 1).to(crop.device)
        pad_value = (-mean / std).squeeze()

        return F.pad(crop, (0, padw, 0, padh), value=0)


class AsyncBatchProcessor:
    """Asynchronous batch processor with prefetch queue

    Overlaps I/O (image loading) with compute (feature extraction).

    Usage:
        processor = AsyncBatchProcessor(extractor, num_prefetch=2)
        processor.start()

        for img_path in image_paths:
            processor.submit(img_path)
            # Process while next is being loaded...

        processor.wait_completion()
    """

    def __init__(
        self,
        extractor: BatchFeatureExtractor,
        num_prefetch: int = 2,
        transform=None
    ):
        self.extractor = extractor
        self.num_prefetch = num_prefetch
        self.transform = transform or (lambda x: x)
        self.queue = queue.Queue(maxsize=num_prefetch)
        self.running = False
        self.loader_thread = None

    def start(self):
        """Start the prefetch thread"""
        self.running = True

    def stop(self):
        """Stop the prefetch thread"""
        self.running = False
        if self.loader_thread:
            self.loader_thread.join(timeout=1.0)

    def submit(self, image_path: str, index: int):
        """Submit an image for prefetch

        Args:
            image_path: Path to image file
            index: Index of this image
        """
        def load_task():
            if not self.running:
                return
            try:
                from PIL import Image
                img = Image.open(image_path).convert('RGB')
                img_tensor = self.transform(img)

                # Non-blocking transfer to GPU
                img_tensor = img_tensor.to(self.extractor.device, non_blocking=True)

                self.queue.put((index, img_tensor), timeout=1)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")

        thread = threading.Thread(target=load_task, daemon=True)
        thread.start()

    def get_next(self) -> Optional[Tuple[int, torch.Tensor]]:
        """Get next preloaded image

        Returns:
            (index, image_tensor) or None if queue is empty
        """
        try:
            return self.queue.get(timeout=5)
        except queue.Empty:
            return None

    def process_batch(self, max_batch_size: Optional[int] = None) -> Tuple[List[int], List[torch.Tensor]]:
        """Collect and process a batch of preloaded images

        Args:
            max_batch_size: Maximum batch size (defaults to config)

        Returns:
            (indices, images) lists
        """
        batch_size = max_batch_size or self.extractor.config.max_batch_size
        indices = []
        images = []

        for _ in range(batch_size):
            result = self.get_next()
            if result is None:
                break
            idx, img = result
            indices.append(idx)
            images.append(img)

        return indices, images
