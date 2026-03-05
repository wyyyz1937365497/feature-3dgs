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

"""Semantic feature dataparser that extends NerfstudioDataparser with semantic feature support."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    Nerfstudio,
    NerfstudioDataParserConfig,
)
from nerfstudio.data.scene_box import SceneBox
from rich.console import Console

CONSOLE = Console(width=120)


@dataclass
class SemanticFeatureDataparserConfig(NerfstudioDataParserConfig):
    """Semantic feature dataparser configuration.

    Extends NerfstudioDataParserConfig with semantic feature support.
    """

    _target: Type = field(default_factory=lambda: SemanticFeatureDataparser)

    # Semantic feature specific configs
    semantic_feature_dir: Optional[str] = None
    """Directory containing pre-computed semantic feature .pt files."""
    semantic_feature_dim: int = 512
    """Dimension of the semantic features."""
    use_speedup: bool = False
    """If True, use compressed features (1/4 dimension) for faster training."""


@dataclass
class SemanticFeatureDataparserOutputs(DataparserOutputs):
    """Semantic feature dataparser outputs with semantic feature metadata.

    Extends DataparserOutputs to include semantic_features.
    """

    # Semantic feature metadata
    semantic_feature_dim: int = 512
    semantic_features: Optional[Dict[str, torch.Tensor]] = None
    """Mapping from image filename to semantic feature tensor."""


class SemanticFeatureDataparser(Nerfstudio):
    """Semantic feature dataparser that extends Nerfstudio.

    This dataparser loads pre-computed semantic features for each training image
    and includes them in the metadata for the model to use during training.
    """

    config: SemanticFeatureDataparserConfig

    def __init__(self, config: SemanticFeatureDataparserConfig):
        super().__init__(config)
        self.semantic_features: Dict[str, torch.Tensor] = {}

    def _generate_dataparser_outputs(
        self, split: str = "train", **kwargs
    ) -> SemanticFeatureDataparserOutputs:
        """Generate dataparser outputs with semantic features.

        Args:
            split: Which split to generate (train, val, test).

        Returns:
            SemanticFeatureDataparserOutputs containing images, cameras, and semantic features.
        """
        # Get base outputs from parent class
        base_outputs = super()._generate_dataparser_outputs(split=split, **kwargs)

        # Load semantic features if directory is specified
        semantic_features = None
        if self.config.semantic_feature_dir is not None:
            semantic_features = self._load_semantic_features(
                base_outputs.image_filenames, self.config.semantic_feature_dir
            )
            CONSOLE.print(
                f"[bold green]Loaded {len(semantic_features)} semantic features for {split} split[/bold green]"
            )

        # Adjust feature dimension if using speedup mode
        feature_dim = self.config.semantic_feature_dim
        if self.config.use_speedup:
            feature_dim = feature_dim // 4
            CONSOLE.print(
                f"[bold yellow]Using speedup mode: feature dimension reduced to {feature_dim}[/bold yellow]"
            )

        return SemanticFeatureDataparserOutputs(
            image_filenames=base_outputs.image_filenames,
            cameras=base_outputs.cameras,
            scene_box=base_outputs.scene_box,
            dataparser_transform=base_outputs.dataparser_transform,
            metadata=base_outputs.metadata,
            # Semantic feature specific fields
            semantic_feature_dim=feature_dim,
            semantic_features=semantic_features,
        )

    def _load_semantic_features(
        self, image_filenames: List[Path], feature_dir: str
    ) -> Dict[str, torch.Tensor]:
        """Load pre-computed semantic features from .pt files.

        Args:
            image_filenames: List of image file paths.
            feature_dir: Directory containing the .pt feature files.

        Returns:
            Dictionary mapping image filenames to feature tensors.
        """
        feature_dir_path = Path(feature_dir)
        if not feature_dir_path.exists():
            CONSOLE.print(
                f"[bold red]Warning: Semantic feature directory {feature_dir} not found. "
                "Semantic features will not be used.[/bold red]"
            )
            return {}

        semantic_features = {}
        loaded_count = 0

        for img_path in image_filenames:
            # Find corresponding feature file
            # Feature files are named like: <image_name>.pt or <image_name>_feature.pt
            img_name = img_path.stem
            feature_path = feature_dir_path / f"{img_name}.pt"

            if not feature_path.exists():
                # Try alternative naming
                feature_path = feature_dir_path / f"{img_name}_feature.pt"

            if feature_path.exists():
                try:
                    feature = torch.load(feature_path, map_location="cpu")
                    # Ensure feature is a tensor
                    if not isinstance(feature, torch.Tensor):
                        CONSOLE.print(
                            f"[bold yellow]Warning: Feature file {feature_path} does not contain a tensor. Skipping.[/bold yellow]"
                        )
                        continue

                    # Store feature with image filename as key
                    semantic_features[str(img_path)] = feature
                    loaded_count += 1
                except Exception as e:
                    CONSOLE.print(
                        f"[bold red]Error loading feature from {feature_path}: {e}[/bold red]"
                    )
            else:
                CONSOLE.print(
                    f"[bold yellow]Warning: No feature file found for {img_name} at {feature_path}[/bold yellow]"
                )

        CONSOLE.print(
            f"[green]Loaded {loaded_count}/{len(image_filenames)} semantic features[/green]"
        )
        return semantic_features


def get_config(data: Path, semantic_feature_dir: Optional[str] = None) -> SemanticFeatureDataparserConfig:
    """Get a SemanticFeatureDataparserConfig for the given data directory.

    Args:
        data: Path to the data directory.
        semantic_feature_dir: Path to the directory containing pre-computed semantic features.

    Returns:
        SemanticFeatureDataparserConfig instance.
    """
    return SemanticFeatureDataparserConfig(
        data=data,
        semantic_feature_dir=semantic_feature_dir,
    )
