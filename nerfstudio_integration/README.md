# feature-3dgs Nerfstudio Integration

Integration of feature-3dgs into the nerfstudio framework for efficient 3D Gaussian Splatting with semantic features.

## Features

- **Semantic Feature Rendering**: Render N-D semantic features alongside RGB
- **Text-Guided Editing**: Delete, extract, or recolor objects based on semantic similarity
- **Speedup Mode**: Optional CNN decoder for faster training with compressed features
- **Nerfstudio Integration**: Leverage nerfstudio's modular pipeline and tooling

## Installation

1. Install nerfstudio:
```bash
pip install nerfstudio
```

2. Install dependencies:
```bash
pip install gsplat>=1.0.0
pip install segment-anything  # For SAM features
```

3. Set up the LSeg encoder (if using LSeg features):
```bash
cd encoders/lseg_encoder
pip install -r requirements.txt
```

## Directory Structure

```
feature-3dgs/
├── feature_3dgs_extension/        # Extension module
│   ├── models/
│   │   └── feature_3dgs.py       # Core model implementation
│   ├── data/
│   │   ├── dataparsers/
│   │   │   └── semantic_feature_dataparser.py
│   │   └── datasets/
│   │       └── semantic_feature_dataset.py
│   └── configs/
│       └── feature_3dgs_configs.py
├── third_party/
│   └── nerfstudio/               # Git submodule (nerfstudio reference)
├── scripts/                       # Utility scripts
│   ├── precompute_semantic_features.py
│   ├── register_feature_3dgs.py
│   ├── test_integration.py
│   └── editing_demo.py
└── nerfstudio_integration/        # Documentation
```

## Quick Start

### 1. Pre-compute Semantic Features

First, extract semantic features from your images using LSeg or SAM:

```bash
python scripts/precompute_semantic_features.py \
    --data path/to/your/dataset \
    --output path/to/features \
    --model lseg \
    --resize 480 640
```

This creates `.pt` files containing the semantic features for each image.

### 2. Train the Model

#### Option A: Using nerfstudio directly

Register the configs first in your nerfstudio installation:

```python
# Add to nerfstudio/configs/method_configs.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path("path/to/feature-3dgs")))

from feature_3dgs_extension.configs.feature_3dgs_configs import register_feature_3dgs_configs

register_feature_3dgs_configs(method_configs, descriptions)
```

Or use the provided script:

```bash
python scripts/register_feature_3dgs.py
```

Then train:

```bash
ns-train feature-3dgs \
    --data path/to/dataset \
    --semantic-feature-dir path/to/features
```

#### Option B: Using the training script

```bash
python nerfstudio_integration/train_feature_3dgs.py \
    --data path/to/dataset \
    --semantic-features path/to/features \
    --output outputs/my_model
```

### 3. Speedup Mode (Faster Training)

```bash
ns-train feature-3dgs-speedup \
    --data path/to/dataset \
    --semantic-feature-dir path/to/features
```

## Configuration Options

### Model Config (`Feature3DGSModelConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `semantic_feature_dim` | int | 512 | Dimension of semantic features |
| `use_semantic_features` | bool | True | Enable semantic feature rendering |
| `semantic_loss_weight` | float | 1.0 | Weight for semantic loss |
| `use_speedup` | bool | False | Enable CNN decoder speedup |
| `enable_editing` | bool | True | Enable text-guided editing |
| `edit_score_threshold` | float | 0.5 | Similarity threshold for editing |

### Dataparser Config (`SemanticFeatureDataparserConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `semantic_feature_dir` | str | None | Directory of .pt feature files |
| `semantic_feature_dim` | int | 512 | Expected feature dimension |
| `use_speedup` | bool | False | Use compressed features |

## Text-Guided Editing

After training, you can use text features to edit the scene:

```python
from nerfstudio.models.feature_3dgs import Feature3DGSModel

# Load model
model = Feature3DGSModel.load_from_checkpoint("path/to/checkpoint")

# Prepare text feature (e.g., from CLIP or LSeg)
text_feature = extract_text_feature("chair")

# Define edit operation
edit_dict = {
    "positive_ids": [0],  # Which text features to match
    "score_threshold": 0.5,
    "operations": ["deletion"]  # or "extraction" or "color_func"
}

# Render with editing
outputs = model.render_edit(
    camera=camera,
    text_feature=text_feature,
    edit_dict=edit_dict
)
```

### Supported Edit Operations

- **deletion**: Remove matching objects (set opacity to 0)
- **extraction**: Keep only matching objects (remove others)
- **color_func**: Apply color transformation to matching objects

## API Reference

### Feature3DGSModel

Main model class extending `SplatfactoModel`.

#### Methods

- `get_outputs(camera)`: Render RGB + semantic features
- `get_loss_dict(outputs, batch)`: Compute losses including semantic loss
- `render_edit(camera, text_feature, edit_dict)`: Render with text-guided editing
- `_calculate_selection_score(features, query_features)`: Compute semantic similarity

### SemanticFeatureDataparser

Dataparser for loading semantic features.

#### Parameters

- `semantic_feature_dir`: Path to directory with .pt files
- `semantic_feature_dim`: Expected feature dimension
- `use_speedup`: Whether to use compressed features

## Examples

See `examples/` directory for complete examples:
- `basic_training.py`: Simple training script
- `editing_demo.py`: Text-guided editing demo
- `custom_dataset.py`: Using custom datasets

## Troubleshooting

### Out of Memory

- Reduce `semantic_feature_dim` or use `use_speedup=True`
- Reduce batch size or image resolution
- Use gradient checkpointing

### NaN Loss

- Check semantic feature dimensions match
- Verify feature files are correctly formatted
- Reduce learning rates

### Slow Training

- Enable `use_speedup=True` for CNN decoder
- Reduce `semantic_loss_weight`
- Use fewer semantic features

## Citation

If you use this code, please cite the original feature-3dgs and nerfstudio papers:

```bibtex
@article{feature-3dgs,
  title={Feature-3DGS: 3D Gaussian Splatting with Semantic Features},
  author={...},
  year={2024}
}

@inproceedings{nerfstudio,
  title={Nerfstudio: A Modular Framework for NeRFs},
  author={...},
  year={2023}
}
```

## License

Apache 2.0 - See LICENSE file for details.
