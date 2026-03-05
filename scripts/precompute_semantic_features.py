#!/usr/bin/env python
"""
Script to pre-compute semantic features for images using LSeg or SAM models.

This script extracts semantic features from images and saves them as .pt files
that can be used with the feature-3dgs model.

Usage:
    python precompute_semantic_features.py --data <path> --output <dir> --model lseg
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np


def load_lseg_model(device: str = "cuda"):
    """Load the LSeg model for feature extraction.

    Args:
        device: Device to load the model on.

    Returns:
        LSeg model and transform.
    """
    try:
        from encoders.lseg_encoder.encode_images import LSegEncoder
        model = LSegEncoder(device=device)
        transform = model.get_transform()
        return model, transform
    except ImportError:
        print("LSeg encoder not found. Please ensure encoders/lseg_encoder is available.")
        return None, None


def load_sam_model(device: str = "cuda"):
    """Load the SAM model for feature extraction.

    Args:
        device: Device to load the model on.

    Returns:
        SAM model and transform.
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        # Load SAM model - you may need to adjust the model path
        model_path = "path/to/sam_vit_h.pth"  # Update this path
        sam = sam_model_registry["vit_h"](checkpoint=model_path)
        sam.to(device=device)
        return sam, None
    except ImportError:
        print("SAM not found. Please install segment-anything package.")
        return None, None


def extract_features_lseg(
    image_paths: List[Path],
    model,
    transform,
    device: str = "cuda",
    target_size: Optional[tuple] = None,
) -> dict:
    """Extract features using LSeg model.

    Args:
        image_paths: List of image paths.
        model: LSeg model.
        transform: Image transform.
        device: Device to use.
        target_size: Target size for resizing (height, width). None for original size.

    Returns:
        Dictionary mapping image names to features.
    """
    features_dict = {}
    model.eval()

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Extracting LSeg features"):
            img = Image.open(img_path).convert("RGB")

            # Resize if target size is specified
            if target_size is not None:
                img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)

            img_tensor = transform(img).unsqueeze(0).to(device)

            # Extract features
            features = model.encode_images(img_tensor)  # [1, C, H, W]

            # Convert to [H, W, C] format
            features = features.squeeze(0).permute(1, 2, 0).cpu()  # [H, W, C]

            features_dict[img_path.stem] = features

    return features_dict


def extract_features_sam(
    image_paths: List[Path],
    model,
    device: str = "cuda",
    target_size: Optional[tuple] = None,
) -> dict:
    """Extract features using SAM model (segmentation-based).

    Args:
        image_paths: List of image paths.
        model: SAM model.
        device: Device to use.
        target_size: Target size for resizing.

    Returns:
        Dictionary mapping image names to features.
    """
    from segment_anything import SamAutomaticMaskGenerator

    mask_generator = SamAutomaticMaskGenerator(model)
    features_dict = {}

    for img_path in tqdm(image_paths, desc="Extracting SAM features"):
        img = np.array(Image.open(img_path).convert("RGB"))

        # Resize if target size is specified
        if target_size is not None:
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray(img)
            img_pil = img_pil.resize((target_size[1], target_size[0]), PILImage.BILINEAR)
            img = np.array(img_pil)

        # Generate masks
        masks = mask_generator.generate(img)

        # Convert masks to feature tensor
        # This is a simplified version - you may want to use embeddings directly
        H, W = img.shape[:2]
        feature_dim = 256  # Adjust based on your needs

        # Create a feature tensor from mask information
        feature_tensor = torch.zeros(H, W, feature_dim)

        # Here you would process the masks to create meaningful features
        # For now, this is a placeholder

        features_dict[img_path.stem] = feature_tensor

    return features_dict


def save_features(features_dict: dict, output_dir: Path):
    """Save features to disk.

    Args:
        features_dict: Dictionary of features.
        output_dir: Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_name, features in tqdm(features_dict.items(), desc="Saving features"):
        output_path = output_dir / f"{img_name}.pt"
        torch.save(features, output_path)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute semantic features for feature-3dgs")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for features")
    parser.add_argument(
        "--model",
        type=str,
        choices=["lseg", "sam"],
        default="lseg",
        help="Model to use for feature extraction",
    )
    parser.add_argument("--images", type=str, default="images", help="Images subdirectory name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--resize", type=int, nargs=2, default=None, help="Resize images to H W")
    parser.add_argument("--extension", type=str, default=["jpg", "png", "jpeg"], nargs="+", help="Image extensions")

    args = parser.parse_args()

    # Find all images
    data_dir = Path(args.data)
    images_dir = data_dir / args.images
    if not images_dir.exists():
        images_dir = data_dir  # Try data directory directly

    image_paths = []
    for ext in args.extension:
        image_paths.extend(images_dir.glob(f"*.{ext}"))
        image_paths.extend(images_dir.glob(f"*.{ext.upper()}"))

    image_paths = sorted(set(image_paths))
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found!")
        return

    # Load model
    print(f"Loading {args.model.upper()} model...")
    if args.model == "lseg":
        model, transform = load_lseg_model(args.device)
        if model is None:
            return
    else:  # sam
        model, transform = load_sam_model(args.device)
        if model is None:
            return

    # Set target size
    target_size = tuple(args.resize) if args.resize else None

    # Extract features
    print(f"Extracting features from {len(image_paths)} images...")
    if args.model == "lseg":
        features_dict = extract_features_lseg(image_paths, model, transform, args.device, target_size)
    else:
        features_dict = extract_features_sam(image_paths, model, args.device, target_size)

    # Save features
    print(f"Saving features to {args.output}...")
    save_features(features_dict, Path(args.output))

    print(f"Done! Saved {len(features_dict)} feature files.")


if __name__ == "__main__":
    main()
