#!/usr/bin/env python
"""
Simple LSeg feature extraction script for feature-3dgs.

Usage:
    python extract_lseg_features.py --images data/room0/images --output data/room0/features
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Extract LSeg features")
    parser.add_argument("--images", type=str, required=True, help="Images directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--weights", type=str, default="encoders/lseg_encoder/demo_e200.ckpt", help="LSeg weights")
    parser.add_argument("--resize", type=int, nargs=2, default=None, help="Resize to H W")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # 获取项目根目录
    project_root = Path.cwd().resolve()
    lseg_encoder_dir = (project_root / "encoders" / "lseg_encoder").resolve()
    weights_path = (project_root / args.weights).resolve()

    images_dir = (project_root / args.images).resolve()
    output_dir = (project_root / args.output).resolve()

    if not weights_path.exists():
        print(f"[bold red]Error: Weights not found at {weights_path}[/bold red]")
        return

    if not images_dir.exists():
        print(f"[bold red]Error: Images directory not found at {images_dir}[/bold red]")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 添加 lseg_encoder 到路径（确保可以找到内部的 data 模块）
    lseg_str = str(lseg_encoder_dir)
    if lseg_str not in sys.path:
        sys.path.insert(0, lseg_str)

    # 同时也将 lseg_encoder 的父目录添加到路径，以便查找 encoding 等依赖
    parent_dir = str(lseg_encoder_dir.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    print(f"Loading LSeg model from {weights_path}...")

    # 加载 LSeg 模型
    from modules.lseg_module import LSegModule

    # 注意：demo_e200.ckpt 是用 num_features=256 训练的
    # 如果需要 512 维特征，可以后续添加线性投影层
    module = LSegModule.load_from_checkpoint(
        checkpoint_path=str(weights_path),
        data_path=str(lseg_encoder_dir),
        dataset="ignore",
        backbone="clip_vitl16_384",
        num_features=256,  # 必须与 checkpoint 一致
        aux=False,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=255,
        scale_inv=False,
        widehead=True,
        widehead_hr=False,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )

    module = module.to(args.device)
    module.eval()

    # 获取所有图像
    image_paths = []
    for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]:
        image_paths.extend(images_dir.glob(ext))

    image_paths = sorted(set(image_paths))
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found!")
        return

    # 准备 transform
    from torchvision import transforms

    transform_list = [
        transforms.Resize((512, 512), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    transform = transforms.Compose(transform_list)

    target_size = tuple(args.resize) if args.resize else None

    # 提取特征
    print(f"Extracting features...")
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            feature_path = output_dir / f"{img_path.stem}.pt"

            # 跳过已存在的特征
            if feature_path.exists():
                continue

            # 加载图像
            img = Image.open(img_path).convert("RGB")

            # 调整大小
            if target_size is not None:
                img_resized = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
            else:
                img_resized = img.resize((512, 512), Image.BILINEAR)

            # 预处理
            img_tensor = transform(img_resized).unsqueeze(0).to(args.device)

            # 提取特征
            # LSegModule.net 是 LSegNet，调用 forward(x, return_feature=True) 返回特征
            features = module.net(img_tensor, return_feature=True)  # [1, C, H, W]

            # 插值到目标大小
            if target_size is not None:
                features = F.interpolate(
                    features,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                # 使用原始图像大小
                features = F.interpolate(
                    features,
                    size=(img.size[1], img.size[0]),
                    mode='bilinear',
                    align_corners=False
                )

            # 转换为 [H, W, C] 格式
            features = features.squeeze(0).permute(1, 2, 0).cpu()

            # 保存特征
            torch.save(features, feature_path)

    print(f"Done! Features saved to {output_dir}")


if __name__ == "__main__":
    main()
