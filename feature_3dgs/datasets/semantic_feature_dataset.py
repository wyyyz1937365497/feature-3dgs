# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Semantic feature dataset that extends the standard image dataset with semantic features."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from nerfstudio.data.datasets.image_dataset import ImageDataset
from rich.console import Console

CONSOLE = Console(width=120)


class SemanticFeatureDataset(ImageDataset):
    """Dataset that extends ImageDataset with semantic feature support.

    This dataset loads images along with their corresponding semantic features,
    which are pre-computed using models like LSeg or SAM.

    Args:
        image_filenames: List of image file paths.
        semantic_features: Dictionary mapping image filenames to feature tensors.
        **kwargs: Additional arguments passed to ImageDataset.
    """

    def __init__(
        self,
        image_filenames: list,
        semantic_features: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ):
        super().__init__(image_filenames=image_filenames, **kwargs)
        self.semantic_features = semantic_features or {}
        CONSOLE.print(
            f"[green]Initialized SemanticFeatureDataset with {len(self.semantic_features)} feature maps[/green]"
        )

    def get_metadata(self, image_idx: int) -> Dict[str, torch.Tensor]:
        """Get metadata for the given image index, including semantic features.

        Args:
            image_idx: Index of the image to get metadata for.

        Returns:
            Dictionary containing metadata including semantic features if available.
        """
        metadata = super().get_metadata(image_idx)

        # Add semantic feature if available
        image_path = str(self.image_filenames[image_idx])
        if image_path in self.semantic_features:
            feature = self.semantic_features[image_path]

            # Ensure feature is on the correct device and has correct dtype
            if isinstance(feature, torch.Tensor):
                feature = feature.to(dtype=torch.float32)

                # Handle different feature shapes
                # Expected: [H, W, D] or [D, H, W]
                if feature.dim() == 3:
                    # If shape is [D, H, W], permute to [H, W, D]
                    if feature.shape[0] < feature.shape[1] and feature.shape[0] < feature.shape[2]:
                        feature = feature.permute(1, 2, 0)

                metadata["semantic_feature"] = feature
            else:
                CONSOLE.print(
                    f"[bold yellow]Warning: Semantic feature for {image_path} is not a tensor. Skipping.[/bold yellow]"
                )

        return metadata

    def __getitem__(self, image_idx: int) -> Dict[str, torch.Tensor]:
        """Get item with semantic feature.

        Args:
            image_idx: Index of the image to retrieve.

        Returns:
            Dictionary containing the image and its metadata (including semantic feature).
        """
        item = super().__getitem__(image_idx)
        return item


class SemanticFeatureDepthDataset(SemanticFeatureDataset):
    """Dataset that includes both semantic features and depth data.

    Extends SemanticFeatureDataset to also support depth image loading.

    Args:
        image_filenames: List of image file paths.
        depth_filenames: List of depth file paths.
        semantic_features: Dictionary mapping image filenames to feature tensors.
        **kwargs: Additional arguments passed to parent classes.
    """

    def __init__(
        self,
        image_filenames: list,
        depth_filenames: Optional[list] = None,
        semantic_features: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ):
        # Initialize parent class with semantic features
        super().__init__(image_filenames=image_filenames, semantic_features=semantic_features, **kwargs)
        self.depth_filenames = depth_filenames

    def get_metadata(self, image_idx: int) -> Dict[str, torch.Tensor]:
        """Get metadata including depth if available.

        Args:
            image_idx: Index of the image.

        Returns:
            Dictionary containing semantic feature and depth if available.
        """
        metadata = super().get_metadata(image_idx)

        # Add depth if available
        if self.depth_filenames is not None:
            depth_path = self.depth_filenames[image_idx]
            if depth_path is not None and Path(depth_path).exists():
                depth = self.load_depth(depth_path)
                metadata["depth"] = depth

        return metadata

    @staticmethod
    def load_depth(depth_path: Path) -> torch.Tensor:
        """Load depth image from file.

        Args:
            depth_path: Path to the depth image file.

        Returns:
            Depth tensor.
        """
        # Implement depth loading logic based on file format
        # This is a placeholder - actual implementation depends on depth file format
        from PIL import Image

        depth_image = Image.open(depth_path)
        depth = torch.from_numpy(
            numpy.array(depth_image, dtype=numpy.float32) / 65535.0
        )  # Normalize to [0, 1]
        return depth


def create_semantic_feature_dataset(
    dataparser_outputs,
    scale_factor: float = 1.0,
) -> SemanticFeatureDataset:
    """Create a SemanticFeatureDataset from dataparser outputs.

    Args:
        dataparser_outputs: DataparserOutputs containing image filenames and metadata.
        scale_factor: Scale factor for resizing images.

    Returns:
        SemanticFeatureDataset instance.
    """
    # Extract semantic features from metadata if available
    semantic_features = None
    if hasattr(dataparser_outputs, "semantic_features"):
        semantic_features = dataparser_outputs.semantic_features

    return SemanticFeatureDataset(
        image_filenames=dataparser_outputs.image_filenames,
        semantic_features=semantic_features,
        scale_factor=scale_factor,
    )
