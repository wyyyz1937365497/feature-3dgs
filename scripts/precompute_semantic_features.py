#!/usr/bin/env python
"""
Script to pre-compute semantic features for images using LSeg or SAM models.

Directly calls official encoder scripts via subprocess.

Usage:
    python precompute_semantic_features.py --data <path> --output <dir> --model lseg
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm


def extract_lseg_features(
    image_paths: List[Path],
    output_dir: Path,
    weights: Optional[Path] = None,
    device: str = "auto",
    optimize_level: int = 2,
    batch_size: int = 4,
) -> dict:
    """使用官方 LSeg 脚本提取特征

    官方命令（来自 README.md）：
    cd encoders/lseg_encoder
    python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir ../../data/DATASET_NAME/rgb_feature_langseg --test-rgb-dir ../../data/DATASET_NAME/images --workers 0

    Optimization Levels:
        0: Original (no optimization)
        1: Phase 1 optimizations (CPU-GPU transfer, tensor caching, vectorized grid)
        2: Phase 1 + 2 optimizations (batch processing, async I/O, data preloading)
        3: All optimizations including TorchScript compilation

    Args:
        image_paths: 图像路径列表
        output_dir: 输出目录
        weights: 模型权重路径
        device: 设备
        optimize_level: 优化等级 (0-3)
        batch_size: 批处理大小 (level >= 2)

    Returns:
        特征字典
    """
    project_root = Path.cwd().resolve()

    # 设置路径
    lseg_encoder_dir = (project_root / "encoders" / "lseg_encoder").resolve()
    images_dir = image_paths[0].parent.resolve()

    if weights is None:
        weights = Path("encoders/lseg_encoder/demo_e200.ckpt")
    else:
        weights = Path(weights).resolve()

    # 创建符号链接到 lseg_encoder，使官方脚本能找到正确的路径
    temp_data_dir = project_root / "temp_lseg_data"

    # 清理可能存在的旧临时目录
    if temp_data_dir.exists():
        try:
            shutil.rmtree(temp_data_dir)
        except Exception as e:
            print(f"Warning: Could not remove old temp directory: {e}")

    temp_data_dir.mkdir(parents=True, exist_ok=True)
    temp_images_dir = temp_data_dir / "images"
    temp_images_dir.mkdir(parents=True, exist_ok=True)

    # 复制或链接图像到临时目录
    for img_path in image_paths:
        dest_path = temp_images_dir / img_path.name

        # 创建符号链接或复制文件
        try:
            dest_path.symlink_to(img_path.resolve())
        except (OSError, NotImplementedError):
            # Windows 可能不支持符号链接，使用复制
            try:
                shutil.copy(img_path, dest_path)
            except shutil.SameFileError:
                # 源文件和目标文件是同一个文件，跳过
                pass

    # 构建命令 - 支持优化等级
    cmd = [
        sys.executable,
        "-u",
        str(lseg_encoder_dir / "encode_images.py"),
        "--backbone", "clip_vitl16_384",
        "--weights", str(weights),
        "--widehead",
        "--no-scaleinv",
        "--outdir", str(output_dir.resolve()),
        "--test-rgb-dir", str(temp_images_dir),
        "--workers", "0",
        "--optimize-level", str(optimize_level),
    ]

    # 添加批处理大小参数（level >= 2）
    if optimize_level >= 2:
        cmd.extend(["--batch-size", str(batch_size)])

    print(f"Running LSeg feature extraction with optimization level {optimize_level}...")
    print(f"Command: {' '.join(cmd)}")

    # 切换到 lseg_encoder 目录运行（官方脚本需要在正确的目录下运行）
    result = subprocess.run(
        cmd,
        cwd=str(lseg_encoder_dir),
        env=os.environ.copy(),
        check=False
    )

    # 清理临时目录
    shutil.rmtree(temp_data_dir)

    if result.returncode != 0:
        print(f"[Error] LSeg extraction failed with code {result.returncode}")
        return {}

    # 加载生成的特征（官方格式）
    features_dict = {}
    output_dir = Path(output_dir).resolve()
    for img_path in tqdm(image_paths, desc="Loading LSeg features"):
        # 官方脚本生成 *_fmap_CxHxW.pt 文件
        feature_path = output_dir / f"{img_path.stem}_fmap_CxHxW.pt"
        if feature_path.exists():
            try:
                features = torch.load(feature_path)
                features_dict[str(img_path)] = features
            except Exception as e:
                print(f"Warning: Failed to load {feature_path}: {e}")

    print(f"Loaded {len(features_dict)} feature files")
    return features_dict


def extract_sam_features(
    image_paths: List[Path],
    output_dir: Path,
    checkpoint: Optional[Path] = None,
    model_type: str = "vit_h",
    device: str = "cuda",
) -> dict:
    """使用官方 SAM 脚本提取特征

    官方命令（来自 README.md）：
    cd encoders/sam_encoder
    python export_image_embeddings.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --input ../../data/DATASET_NAME/images --output ../../data/OUTPUT_NAME/sam_embeddings

    Args:
        image_paths: 图像路径列表
        output_dir: 输出目录
        checkpoint: SAM 模型权重路径
        model_type: SAM 模型类型
        device: 设备

    Returns:
        特征字典
    """
    project_root = Path.cwd().resolve()

    # 设置路径
    sam_encoder_dir = (project_root / "encoders" / "sam_encoder").resolve()
    images_dir = image_paths[0].parent.resolve()

    if checkpoint is None:
        checkpoint = Path("encoders/sam_encoder/checkpoints/sam_vit_h_4b8939.pth")
    else:
        checkpoint = Path(checkpoint).resolve()

    if not checkpoint.exists():
        print(f"[Error] SAM checkpoint not found at {checkpoint}")
        print("  Please download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return {}

    # 创建符号链接，使官方脚本能找到正确的路径
    temp_data_dir = project_root / "temp_sam_data"

    # 清理可能存在的旧临时目录
    if temp_data_dir.exists():
        try:
            shutil.rmtree(temp_data_dir)
        except Exception as e:
            print(f"Warning: Could not remove old temp directory: {e}")

    temp_data_dir.mkdir(parents=True, exist_ok=True)
    temp_images_dir = temp_data_dir / "images"
    temp_images_dir.mkdir(parents=True, exist_ok=True)

    # 复制或链接图像到临时目录
    for img_path in image_paths:
        dest_path = temp_images_dir / img_path.name

        # 创建符号链接或复制文件
        try:
            dest_path.symlink_to(img_path.resolve())
        except (OSError, NotImplementedError):
            # Windows 可能不支持符号链接，使用复制
            try:
                shutil.copy(img_path, dest_path)
            except shutil.SameFileError:
                # 源文件和目标文件是同一个文件，跳过
                pass

    # 构建命令 - 完全按照官方格式
    cmd = [
        sys.executable,
        str(sam_encoder_dir / "export_image_embeddings.py"),
        "--checkpoint", str(checkpoint),
        "--model-type", model_type,
        "--input", str(temp_images_dir),
        "--output", str(output_dir.resolve()),
    ]

    print(f"Running official SAM script...")
    print(f"Command: {' '.join(cmd)}")

    # 切换到 sam_encoder 目录运行（官方脚本需要在正确的目录下运行）
    result = subprocess.run(
        cmd,
        cwd=str(sam_encoder_dir),
        env=os.environ.copy(),
        check=False
    )

    # 清理临时目录
    shutil.rmtree(temp_data_dir)

    if result.returncode != 0:
        print(f"[Error] SAM extraction failed with code {result.returncode}")
        return {}

    # 加载生成的特征（官方格式）
    features_dict = {}
    output_dir = Path(output_dir).resolve()
    for img_path in tqdm(image_paths, desc="Loading SAM features"):
        # SAM 生成 *.pt 文件
        feature_path = output_dir / f"{img_path.stem}.pt"
        if feature_path.exists():
            try:
                features = torch.load(feature_path)
                features_dict[str(img_path)] = features
            except Exception as e:
                print(f"Warning: Failed to load {feature_path}: {e}")

    print(f"Loaded {len(features_dict)} feature files")
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
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cuda:0, cpu)")
    parser.add_argument("--extension", type=str, default=["jpg", "png", "jpeg"], nargs="+", help="Image extensions")

    # Optimization settings
    parser.add_argument(
        "--optimize-level",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="Optimization level: 0=original, 1=quick wins, 2=with batch+async, 3=full optimization"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for optimization level >= 2"
    )

    # LSeg specific
    parser.add_argument("--weights", type=str, default=None, help="Path to LSeg model weights")

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

    # 提取特征
    if args.model == "lseg":
        features_dict = extract_lseg_features(
            image_paths=image_paths,
            output_dir=Path(args.output),
            weights=Path(args.weights) if args.weights else None,
            device=args.device,
            optimize_level=args.optimize_level,
            batch_size=args.batch_size,
        )
    else:  # sam
        features_dict = extract_sam_features(
            image_paths=image_paths,
            output_dir=Path(args.output),
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            model_type=args.model_type,
            device=args.device,
        )

    if features_dict:
        print(f"Done! Saved {len(features_dict)} feature files to {args.output}")
    else:
        print("Feature extraction failed!")


if __name__ == "__main__":
    main()
