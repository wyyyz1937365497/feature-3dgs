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
feature-3dgs integration with nerfstudio.

This package provides the integration of feature-3dgs into nerfstudio framework.
"""

from nerfstudio.models.feature_3dgs import (
    Feature3DGSModel,
    Feature3DGSModelConfig,
    CNNDecoder,
)
from nerfstudio.data.dataparsers.semantic_feature_dataparser import (
    SemanticFeatureDataparser,
    SemanticFeatureDataparserConfig,
    SemanticFeatureDataparserOutputs,
)
from nerfstudio.data.datasets.semantic_feature_dataset import (
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
