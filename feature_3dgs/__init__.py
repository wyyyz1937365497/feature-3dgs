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
feature-3dgs: 3D Gaussian Splatting with semantic feature support.

This package extends nerfstudio with semantic feature rendering and text-guided editing.
"""

__version__ = "0.1.0"

# Import main components
from feature_3dgs.feature_3dgs import (
    Feature3DGSModel,
    Feature3DGSModelConfig,
    CNNDecoder,
)
from feature_3dgs.dataparsers.semantic_feature_dataparser import (
    SemanticFeatureDataparser,
    SemanticFeatureDataparserConfig,
    SemanticFeatureDataparserOutputs,
)
from feature_3dgs.datasets.semantic_feature_dataset import (
    SemanticFeatureDataset,
    SemanticFeatureDepthDataset,
)

__all__ = [
    "Feature3DGSModel",
    "Feature3DGSModelConfig",
    "CNNDecoder",
    "SemanticFeatureDataparser",
    "SemanticFeatureDataparserConfig",
    "SemanticFeatureDataparserOutputs",
    "SemanticFeatureDataset",
    "SemanticFeatureDepthDataset",
]
