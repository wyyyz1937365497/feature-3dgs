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

"""
feature-3dgs method configurations for nerfstudio.

This file contains the trainer configurations for the feature-3dgs model,
which extends Splatfacto with semantic feature support and text-guided editing.
"""

from typing import Dict, Union

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

# Import the model and dataparser configs
import sys
from pathlib import Path

# Add the project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from the extension module
from feature_3dgs_extension.models.feature_3dgs import Feature3DGSModelConfig
from feature_3dgs_extension.data.dataparsers.semantic_feature_dataparser import SemanticFeatureDataparserConfig


def register_feature_3dgs_configs(
    method_configs: Dict[str, Union[TrainerConfig, object]],
    descriptions: Dict[str, str],
) -> None:
    """Register feature-3dgs method configurations.

    This function should be called to add the feature-3dgs configurations
    to the nerfstudio method_configs and descriptions dictionaries.

    Args:
        method_configs: Dictionary to add method configurations to.
        descriptions: Dictionary to add method descriptions to.
    """

    # Add description
    descriptions["feature-3dgs"] = (
        "3D Gaussian Splatting with semantic feature support and text-guided editing. "
        "Supports pre-computed semantic features from LSeg/SAM models."
    )
    descriptions["feature-3dgs-speedup"] = (
        "Feature-3dgs with CNN decoder for faster training using compressed features."
    )

    # Standard feature-3dgs configuration
    method_configs["feature-3dgs"] = TrainerConfig(
        method_name="feature-3dgs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=Feature3DGSModelConfig(
                semantic_feature_dim=512,
                use_semantic_features=True,
                semantic_loss_weight=1.0,
                use_speedup=False,
                enable_editing=True,
                edit_score_threshold=0.5,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "semantic_features": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    )

    # Speedup configuration with CNN decoder
    method_configs["feature-3dgs-speedup"] = TrainerConfig(
        method_name="feature-3dgs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=Feature3DGSModelConfig(
                semantic_feature_dim=512,
                use_semantic_features=True,
                semantic_loss_weight=1.0,
                use_speedup=True,  # Enable CNN decoder for faster training
                enable_editing=True,
                edit_score_threshold=0.5,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "semantic_features": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "cnn_decoder": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    )


# Auto-register when imported
def _auto_register():
    """Auto-register feature-3dgs configs with nerfstudio."""
    try:
        from nerfstudio.configs import method_configs

        if hasattr(method_configs, "method_configs"):
            register_feature_3dgs_configs(
                method_configs.method_configs,
                method_configs.descriptions,
            )
    except ImportError:
        # nerfstudio not available, skip auto-registration
        pass


# Uncomment to enable auto-registration
# _auto_register()
