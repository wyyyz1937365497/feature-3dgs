#!/usr/bin/env python
"""
Demo script for text-guided 3D scene editing with feature-3dgs.

This script demonstrates how to use the render_edit functionality for:
- Object deletion
- Object extraction
- Color modification

Usage:
    python editing_demo.py --checkpoint <path> --text "chair" --operation deletion
"""

import argparse
from pathlib import Path
import sys

import torch
import numpy as np
from PIL import Image


def extract_text_feature(text: str, method: str = "lseg") -> torch.Tensor:
    """Extract semantic feature from text query.

    Args:
        text: Text query (e.g., "chair", "table")
        method: Feature extraction method ("lseg", "clip", or "random")

    Returns:
        Feature tensor of shape [D]
    """
    feature_dim = 512

    if method == "lseg":
        try:
            from encoders.lseg_encoder.encode_images import LSegEncoder
            encoder = LSegEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
            feature = encoder.encode_text(text)
            return feature
        except ImportError:
            print("LSeg encoder not available, using random features")
            return torch.randn(feature_dim)

    elif method == "clip":
        try:
            import clip
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            text_tokens = clip.tokenize([text]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
            return text_features.squeeze(0)
        except ImportError:
            print("CLIP not available, using random features")
            return torch.randn(feature_dim)

    else:  # random
        print(f"Using random features for text: {text}")
        return torch.randn(feature_dim)


def color_grayscale(shs):
    """Color function to convert to grayscale."""
    # shs is [N, 3] RGB features
    gray = shs.mean(dim=-1, keepdim=True).repeat(1, 3)
    return gray


def color_invert(shs):
    """Color function to invert colors."""
    return 1.0 - shs


def color_sepia(shs):
    """Color function to apply sepia tone."""
    # Sepia transformation matrix
    sepia_matrix = torch.tensor([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ], device=shs.device)
    return torch.clamp(shs @ sepia_matrix.T, 0, 1)


def main():
    parser = argparse.ArgumentParser(description="feature-3dgs text-guided editing demo")

    # Model checkpoint
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    # Editing parameters
    parser.add_argument("--text", type=str, required=True, help="Text query for editing")
    parser.add_argument("--operation", type=str, choices=["deletion", "extraction", "color"], default="deletion",
                        help="Editing operation")
    parser.add_argument("--color-func", type=str, choices=["grayscale", "invert", "sepia"], default="grayscale",
                        help="Color transformation (for 'color' operation)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold")

    # Camera/output
    parser.add_argument("--camera-path", type=str, default=None, help="Path to camera .npy file")
    parser.add_argument("--output", type=str, default="edit_output.png", help="Output image path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    # Feature extraction
    parser.add_argument("--feature-method", type=str, choices=["lseg", "clip", "random"], default="random",
                        help="Text feature extraction method")

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    try:
        from feature_3dgs_extension.models.feature_3dgs import Feature3DGSModel

        # This is a simplified loading - adjust based on actual checkpoint format
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = Feature3DGSModel.load_from_checkpoint(args.checkpoint)
        model.eval()
        model.to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the checkpoint path is correct")
        return 1

    # Extract text feature
    print(f"Extracting feature for text: '{args.text}'")
    text_feature = extract_text_feature(args.text, args.feature_method).to(device)
    print(f"Text feature shape: {text_feature.shape}")

    # Setup camera
    if args.camera_path:
        camera_data = np.load(args.camera_path, allow_pickle=True).item()
        # Create camera from data - adjust based on actual camera format
        from nerfstudio.cameras.cameras import Cameras
        camera = Cameras(
            camera_to_worlds=torch.from_numpy(camera_data["camera_to_worlds"]).float().unsqueeze(0).to(device),
            fx=torch.tensor([camera_data["fx"]]).float().unsqueeze(0).to(device),
            fy=torch.tensor([camera_data["fy"]]).float().unsqueeze(0).to(device),
            cx=torch.tensor([camera_data["cx"]]).float().unsqueeze(0).to(device),
            cy=torch.tensor([camera_data["cy"]]).float().unsqueeze(0).to(device),
            width=torch.tensor([camera_data["width"]]).int().unsqueeze(0),
            height=torch.tensor([camera_data["height"]]).int().unsqueeze(0),
        )
    else:
        print("No camera path provided, using default camera")
        # Create a default camera - adjust parameters as needed
        from nerfstudio.cameras.cameras import Cameras
        camera = Cameras(
            camera_to_worlds=torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32).unsqueeze(0).to(device),
            fx=torch.tensor([500.0]).float().unsqueeze(0).to(device),
            fy=torch.tensor([500.0]).float().unsqueeze(0).to(device),
            cx=torch.tensor([400.0]).float().unsqueeze(0).to(device),
            cy=torch.tensor([300.0]).float().unsqueeze(0).to(device),
            width=torch.tensor([800]).int().unsqueeze(0),
            height=torch.tensor([600]).int().unsqueeze(0),
        )

    # Create edit dictionary
    edit_dict = {
        "positive_ids": [0],
        "score_threshold": args.threshold,
        "operations": [args.operation],
    }

    # Add color function if needed
    if args.operation == "color":
        if args.color_func == "grayscale":
            edit_dict["color_func"] = color_grayscale
        elif args.color_func == "invert":
            edit_dict["color_func"] = color_invert
        elif args.color_func == "sepia":
            edit_dict["color_func"] = color_sepia

    print(f"\nPerforming edit operation: {args.operation}")
    print(f"  Text query: '{args.text}'")
    print(f"  Threshold: {args.threshold}")

    # Render with editing
    with torch.no_grad():
        try:
            outputs = model.render_edit(
                camera=camera,
                text_feature=text_feature,
                edit_dict=edit_dict,
            )

            rgb = outputs["rgb"].cpu().numpy()
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

            # Save output
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(output_path)
            print(f"\n[green]Saved edited image to: {output_path}[/green]")

        except Exception as e:
            print(f"[red]Error during rendering: {e}[/red]")
            import traceback
            traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
