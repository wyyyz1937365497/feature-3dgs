#!/usr/bin/env python
"""
Script to pre-compute semantic features for images using LSeg or SAM models.

Usage:
    python precompute_semantic_features.py --data <path> --output <dir> --model lseg
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np


def extract_lseg_features_subprocess(
    image_paths: List[Path],
    output_dir: Path,
    backbone: str = "clip_vitl16_384",
    weights: Optional[Path] = None,
    target_size: Optional[tuple] = None,
    device: str = "cuda",
) -> dict:
    """使用 LSeg 提取特征（使用简化的提取脚本）

    Args:
        image_paths: 图像路径列表
        output_dir: 输出目录
        backbone: LSeg 主干网络
        weights: 模型权重路径
        target_size: 目标大小 (H, W)
        device: 设备

    Returns:
        特征字典
    """
    project_root = Path.cwd().resolve()
    extract_script = project_root / "scripts" / "extract_lseg_features.py"

    if not extract_script.exists():
        print(f"[bold red]Error: Extract script not found at {extract_script}[/bold red]")
        return {}

    # 构建命令
    cmd = [
        sys.executable,
        str(extract_script),
        "--images", str(image_paths[0].parent),
        "--output", str(output_dir),
        "--weights", str(weights) if weights else "encoders/lseg_encoder/demo_e200.ckpt",
    ]

    if target_size is not None:
        cmd.extend(["--resize", str(target_size[0]), str(target_size[1])])

    cmd.extend(["--device", device])

    print(f"Running LSeg feature extraction...")
    print(f"Images: {image_paths[0].parent}")
    print(f"Output: {output_dir}")

    # 运行脚本
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=os.environ.copy(),
        check=False
    )

    if result.returncode != 0:
        print("[bold red]LSeg feature extraction failed![/bold red]")
        return {}

    # 加载生成的特征
    features_dict = {}
    output_dir = Path(output_dir).resolve()
    for img_path in tqdm(image_paths, desc="Loading LSeg features"):
        feature_path = output_dir / f"{img_path.stem}.pt"
        if feature_path.exists():
            try:
                features_dict[str(img_path)] = torch.load(feature_path)
            except:
                pass

    print(f"Loaded {len(features_dict)} feature files")
    return features_dict


def extract_sam_features(
    image_paths: List[Path],
    output_dir: Path,
    checkpoint: Optional[Path] = None,
    model_type: str = "vit_h",
    target_size: Optional[tuple] = None,
    device: str = "cuda",
) -> dict:
    """使用 SAM 提取特征

    Args:
        image_paths: 图像路径列表
        output_dir: 输出目录
        checkpoint: SAM 模型权重路径
        model_type: SAM 模型类型
        target_size: 目标大小 (H, W)
        device: 设备

    Returns:
        特征字典
    """
    project_root = Path.cwd().resolve()

    # 设置默认权重路径
    if checkpoint is None:
        checkpoint = project_root / "encoders" / "sam_encoder" / "checkpoints" / "sam_vit_h_4b8939.pth"
    else:
        checkpoint = Path(checkpoint).resolve()

    if not checkpoint.exists():
        print(f"[bold red]Error: SAM checkpoint not found at {checkpoint}[/bold red]")
        print("  Please download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return {}

    try:
        from segment_anything import sam_model_registry
    except ImportError:
        print("[bold red]Error: SAM not installed[/bold red]")
        print("  Please install: pip install git+https://github.com/facebookresearch/segment-anything.git")
        return {}

    # 加载模型
    print(f"Loading SAM model from {checkpoint}...")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device)
    sam.eval()

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features_dict = {}

    print(f"Extracting features from {len(image_paths)} images...")

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Extracting SAM features"):
            # 检查是否已存在
            feature_path = output_dir / f"{img_path.stem}.pt"
            if feature_path.exists():
                try:
                    features_dict[str(img_path)] = torch.load(feature_path)
                    continue
                except:
                    pass

            # 加载图像
            img = np.array(Image.open(img_path).convert("RGB"))

            # 调整大小（如果指定）
            original_size = tuple(img.shape[:2])
            if target_size is not None:
                from PIL import Image as PILImage
                img_pil = PILImage.fromarray(img)
                img_pil = img_pil.resize((target_size[1], target_size[0]), PILImage.BILINEAR)
                img = np.array(img_pil)

            # SAM 预处理
            from segment_anything.utils.transforms import ResizeLongestSide

            transform = ResizeLongestSide(sam.image_encoder.img_size)
            input_image = transform.apply_image(img)
            input_image_torch = torch.as_tensor(input_image, device=device).permute(2, 0, 1).contiguous()

            # 提取特征
            features = sam.image_encoder(input_image_torch.unsqueeze(0))  # [1, C, H, W]

            # 调整到原始图像大小
            features = F.interpolate(
                features,
                size=(original_size[1], original_size[0]),
                mode='bilinear',
                align_corners=False
            )

            # 转换为 [H, W, C] 格式
            features = features.squeeze(0).permute(1, 2, 0).cpu()  # [H, W, C]

            # 保存特征
            torch.save(features, feature_path)
            features_dict[str(img_path)] = features

    return features_dict


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

    # LSeg specific
    parser.add_argument("--backbone", type=str, default="clip_vitl16_384", help="LSeg backbone")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")

    # SAM specific
    parser.add_argument("--checkpoint", type=str, default=None, help="SAM checkpoint path")
    parser.add_argument("--model-type", type=str, default="vit_h", help="SAM model type")

    args = parser.parse_args()

    # 查找所有图像
    data_dir = Path(args.data)
    images_dir = data_dir / args.images
    if not images_dir.exists():
        images_dir = data_dir  # 尝试直接使用数据目录

    image_paths = []
    for ext in args.extension:
        image_paths.extend(images_dir.glob(f"*.{ext}"))
        image_paths.extend(images_dir.glob(f"*.{ext.upper()}"))

    image_paths = sorted(set(image_paths))
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found!")
        return

    # 确定目标大小
    target_size = tuple(args.resize) if args.resize else None

    # 提取特征
    if args.model == "lseg":
        features_dict = extract_lseg_features_subprocess(
            image_paths=image_paths,
            output_dir=Path(args.output),
            backbone=args.backbone,
            weights=Path(args.weights) if args.weights else None,
            target_size=target_size,
            device=args.device,
        )
    else:  # sam
        features_dict = extract_sam_features(
            image_paths=image_paths,
            output_dir=Path(args.output),
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            model_type=args.model_type,
            target_size=target_size,
            device=args.device,
        )

    if features_dict:
        print(f"Done! Saved {len(features_dict)} feature files to {args.output}")
    else:
        print("Feature extraction failed!")


if __name__ == "__main__":
    main()
