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
Feature-3DGS Dataparser: 整合 COLMAP 处理和语义特征提取的数据解析器

支持：
1. 从原始图像自动运行 COLMAP（使用 nerfstudio 内置逻辑）
2. 从已有 COLMAP 数据加载（跳过匹配）
3. 自动提取 LSeg/SAM 语义特征
4. 返回包含语义特征的 DataparserOutputs

使用方式：
    ns-train feature-3dgs --data data/room0 feature-3dgs-data --feature-model lseg
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import os
import torch
from PIL import Image
from rich.console import Console

from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs

CONSOLE = Console(width=120)


@dataclass
class Feature3DGSDataparserConfig(DataParserConfig):
    """Feature-3DGS 数据解析器配置

    支持从原始图像或已有 COLMAP 数据开始，自动提取语义特征。
    """

    _target: Type = field(default_factory=lambda: Feature3DGSDataparser)

    data: Path = field(default_factory=lambda: Path("data/scene"))
    """数据目录路径"""

    # COLMAP 相关配置
    colmap_executable: Optional[Path] = None
    """COLMAP 可执行文件路径（如果不在 PATH 中）"""

    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    """训练/测试分割方式"""

    train_split_fraction: float = 0.9
    """训练集比例"""

    downscale_factor: Optional[int] = None
    """图像缩放因子（None 自动选择）"""

    # 语义特征相关配置
    feature_model: Literal["lseg", "sam", "none"] = "lseg"
    """语义特征提取模型：lseg、sam 或 none（不使用语义特征）"""

    feature_output_dir: Optional[str] = None
    """特征输出目录（默认为 data/FEATURES）"""

    feature_dim: int = 512
    """语义特征维度（LSeg=512 官方格式, SAM=256）"""

    # LSeg 配置
    lseg_weights: Optional[Path] = None
    """LSeg 模型权重路径（默认为 encoders/lseg_encoder/demo_e200.ckpt）"""

    lseg_backbone: str = "clip_vitl16_384"
    """LSeg 主干网络"""

    # SAM 配置
    sam_checkpoint: Optional[Path] = None
    """SAM 模型权重路径（默认为 encoders/sam_encoder/checkpoints/sam_vit_h_4b8939.pth）"""

    sam_model_type: str = "vit_h"
    """SAM 模型类型"""

    # 特征提取配置
    extract_features: bool = True
    """是否自动提取特征（如果特征已存在则跳过）"""

    feature_resize: Optional[Tuple[int, int]] = None
    """特征提取时的图像大小 (H, W)，None 使用原始大小"""

    device: str = "cuda"
    """特征提取设备"""

    # 跳过已有特征
    skip_existing: bool = True
    """跳过已存在的特征文件"""


class Feature3DGSDataparser(DataParser):
    """Feature-3DGS 数据解析器

    整合 COLMAP 处理和语义特征提取的完整数据管线。
    """

    config: Feature3DGSDataparserConfig

    def __init__(self, config: Feature3DGSDataparserConfig):
        super().__init__(config=config)
        self.semantic_features: Dict[str, torch.Tensor] = {}

    def _generate_dataparser_outputs(self, split="train") -> DataparserOutputs:
        """生成数据解析器输出

        Args:
            split: 数据集分割 ("train" 或 "val")

        Returns:
            DataparserOutputs
        """
        from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
            Nerfstudio,
            NerfstudioDataParserConfig,
        )

        # 步骤 1: 使用 nerfstudio 的 COLMAP 处理
        CONSOLE.print("[bold blue]Step 1/3: Processing COLMAP data...[/bold blue]")

        colmap_config = NerfstudioDataParserConfig(
            data=self.config.data,
            eval_mode=self.config.eval_mode,
            train_split_fraction=self.config.train_split_fraction,
            downscale_factor=self.config.downscale_factor,
        )

        colmap_parser = Nerfstudio(config=colmap_config)
        outputs = colmap_parser._generate_dataparser_outputs(split)

        # 步骤 2: 提取语义特征
        if self.config.feature_model != "none":
            CONSOLE.print(f"[bold blue]Step 2/3: Extracting {self.config.feature_model.upper()} features...[/bold blue]")
            self._extract_semantic_features(outputs)
        else:
            CONSOLE.print("[bold yellow]Step 2/3: Skipping feature extraction (feature_model=none)[/bold yellow]")

        # 步骤 3: 添加特征到输出
        CONSOLE.print("[bold blue]Step 3/3: Preparing final outputs...[/bold blue]")

        # 添加语义特征元数据
        metadata = outputs.metadata.copy() if hasattr(outputs, 'metadata') else {}
        if self.semantic_features:
            metadata["semantic_features"] = self.semantic_features
            metadata["semantic_feature_dim"] = self.config.feature_dim
            metadata["feature_model"] = self.config.feature_model
            CONSOLE.print(f"[green]✓ Loaded {len(self.semantic_features)} semantic feature maps[/green]")

        # 返回 DataparserOutputs
        return DataparserOutputs(
            image_filenames=outputs.image_filenames,
            cameras=outputs.cameras,
            scene_box=outputs.scene_box,
            metadata=metadata,
            dataparser_scale=outputs.dataparser_scale,
            dataparser_transform=outputs.dataparser_transform,
        )

    def _extract_semantic_features(self, outputs: DataparserOutputs) -> None:
        """提取语义特征

        Args:
            outputs: COLMAP 数据解析器输出
        """
        # 确定特征输出目录
        if self.config.feature_output_dir is None:
            feature_dir = self.config.data / "features"
        else:
            feature_dir = Path(self.config.feature_output_dir)

        feature_dir.mkdir(parents=True, exist_ok=True)

        # 获取图像文件名
        image_filenames = outputs.image_filenames

        CONSOLE.print(f"  Data directory: {self.config.data}")
        CONSOLE.print(f"  Feature directory: {feature_dir}")
        CONSOLE.print(f"  Images: {len(image_filenames)}")

        # 根据选择的模型提取特征
        if self.config.feature_model == "lseg":
            self._extract_lseg_features(image_filenames, feature_dir)
        elif self.config.feature_model == "sam":
            self._extract_sam_features(image_filenames, feature_dir)

    def _extract_lseg_features(self, image_filenames: List[Path], feature_dir: Path) -> None:
        """使用 LSeg 提取特征（基于官方 encode_images_visfeature.py）

        Args:
            image_filenames: 图像文件名列表
            feature_dir: 特征输出目录
        """
        import sys
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as transforms
        from tqdm import tqdm

        # 确定 LSeg 权重路径
        if self.config.lseg_weights is None:
            lseg_weights = Path("encoders/lseg_encoder/demo_e200.ckpt")
        else:
            lseg_weights = self.config.lseg_weights

        if not lseg_weights.exists():
            CONSOLE.print(f"[bold red]Error: LSeg weights not found at {lseg_weights}[/bold red]")
            CONSOLE.print("  Please download from: https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing")
            return

        # 获取项目根目录和 lseg_encoder 路径
        project_root = Path.cwd().resolve()
        lseg_encoder_dir = (project_root / "encoders" / "lseg_encoder").resolve()
        weights_path = lseg_weights.resolve() if not lseg_weights.is_absolute() else lseg_weights

        # 添加 lseg_encoder 到路径
        lseg_str = str(lseg_encoder_dir)
        if lseg_str not in sys.path:
            sys.path.insert(0, lseg_str)
        parent_dir = str(lseg_encoder_dir.parent)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)

        # 设置环境变量
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # 导入 LSeg 模块 - 使用官方参数
        try:
            from modules.lseg_module import LSegModule
            from additional_utils.encoding_models import MultiEvalModule as LSeg_MultiEvalModule

            CONSOLE.print(f"  Loading LSeg model from {weights_path}...")

            # 官方参数设置
            module = LSegModule.load_from_checkpoint(
                checkpoint_path=str(weights_path),
                data_path=str(lseg_encoder_dir),
                dataset="ignore",
                backbone="clip_vitl16_384",
                num_features=256,
                aux=False,
                aux_weight=0,
                se_loss=False,
                se_weight=0,
                base_lr=0,
                batch_size=1,
                max_epochs=0,
                ignore_index=255,
                dropout=0.0,
                scale_inv=True,  # 官方设置
                augment=False,
                no_batchnorm=False,
                widehead=True,
                widehead_hr=False,
                arch_option=0,
                strict=True,
                block_depth=0,
                activation="lrelu",
            )

            module = module.to(self.config.device)
            module.eval()

            # 创建 MultiEvalModule - 官方的多尺度设置
            labels = module.get_labels('ade20k')
            num_classes = len(labels)
            scales = [0.75, 1.0, 1.25, 1.75]  # 官方设置

            evaluator = LSeg_MultiEvalModule(
                module.net, num_classes, scales=scales, flip=True
            ).to(self.config.device)
            evaluator.eval()

            CONSOLE.print("  [green]✓ LSeg model loaded[/green]")

        except Exception as e:
            CONSOLE.print(f"[bold red]Error loading LSeg: {e}[/bold red]")
            import traceback
            traceback.print_exc()
            return

        # 默认输出尺寸
        default_h, default_w = 480, 360

        # 提取特征
        for img_path in tqdm(image_filenames, desc="  Extracting LSeg features"):
            # 官方格式特征文件名：{filename}_fmap_CxHxW.pt
            feature_path = feature_dir / f"{img_path.stem}_fmap_CxHxW.pt"

            # 跳过已存在的特征
            if self.config.skip_existing and feature_path.exists():
                try:
                    feature = torch.load(feature_path)
                    self.semantic_features[str(img_path)] = feature
                    continue
                except:
                    pass

            try:
                # 加载图像
                img = Image.open(img_path).convert("RGB")

                # 使用官方的 transform
                input_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
                img_tensor = input_transform(img).unsqueeze(0).to(self.config.device)

                # 如果图像太大，调整大小
                h, w = img_tensor.shape[2], img_tensor.shape[3]
                if w > default_w:
                    scale_factor = default_w / w
                    new_h = int(h * scale_factor)
                    img_tensor = F.interpolate(
                        img_tensor, size=(new_h, default_w),
                        mode="bilinear", align_corners=True
                    )
                    h, w = new_h, default_w

                # 使用官方的 MultiEvalModule 获取特征
                with torch.no_grad():
                    with torch.cuda.device_of(img_tensor):
                        output_features = evaluator(img_tensor, return_feature=True)
                        # output_features: [1, 512, h, w] 经过多尺度平均

                # 插值到目标大小
                if self.config.feature_resize is not None:
                    h_target, w_target = self.config.feature_resize
                else:
                    h_target, w_target = h, w

                fmap = F.interpolate(output_features, size=(h_target, w_target), mode='bilinear', align_corners=False)

                # Normalize - 官方逻辑
                fmap = F.normalize(fmap, dim=1)  # [1, 512, h, w]

                # 转换为 [C, H, W] 并保存为 half
                fmap = fmap[0]  # [512, h, w]
                fmap = fmap.cpu().half()  # 使用 half 精度

                # 保存 - 官方格式
                torch.save(fmap, feature_path)
                self.semantic_features[str(img_path)] = fmap

                # 清理
                del img_tensor, output_features, fmap
                if self.config.device.startswith('cuda'):
                    torch.cuda.empty_cache()

            except Exception as e:
                CONSOLE.print(f"[red]Error processing {img_path.name}: {e}[/red]")
                continue

        CONSOLE.print(f"  [green]✓ Extracted {len(self.semantic_features)} LSeg features[/green]")

    def _extract_sam_features(self, image_filenames: List[Path], feature_dir: Path) -> None:
        """使用 SAM 提取特征

        Args:
            image_filenames: 图像文件名列表
            feature_dir: 特征输出目录
        """
        # 确定 SAM 权重路径
        if self.config.sam_checkpoint is None:
            sam_checkpoint = Path("encoders/sam_encoder/checkpoints/sam_vit_h_4b8939.pth")
        else:
            sam_checkpoint = self.config.sam_checkpoint

        if not sam_checkpoint.exists():
            CONSOLE.print(f"[bold red]Error: SAM checkpoint not found at {sam_checkpoint}[/bold red]")
            CONSOLE.print("  Please download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            return

        # 导入 SAM
        try:
            from segment_anything import sam_model_registry
            from tqdm import tqdm
            from PIL import Image
            import numpy as np

            CONSOLE.print(f"  Loading SAM model from {sam_checkpoint}...")
            sam = sam_model_registry[self.config.sam_model_type](checkpoint=str(sam_checkpoint))
            sam.to(self.config.device)
            CONSOLE.print("  [green]✓ SAM model loaded[/green]")

        except ImportError as e:
            CONSOLE.print(f"[bold red]Error importing SAM: {e}[/bold red]")
            CONSOLE.print("  Please install: pip install git+https://github.com/facebookresearch/segment-anything.git")
            return

        sam.eval()

        # 提取特征
        for img_path in tqdm(image_filenames, desc="  Extracting SAM features"):
            # 确定特征文件路径
            feature_path = feature_dir / f"{img_path.stem}.pt"

            # 跳过已存在的特征
            if self.config.skip_existing and feature_path.exists():
                try:
                    feature = torch.load(feature_path)
                    self.semantic_features[str(img_path)] = feature
                    continue
                except:
                    pass

            # 加载图像
            img = np.array(Image.open(img_path).convert("RGB"))

            # 调整大小
            if self.config.feature_resize is not None:
                from PIL import Image as PILImage
                img_pil = PILImage.fromarray(img)
                img_pil = img_pil.resize(
                    (self.config.feature_resize[1], self.config.feature_resize[0]),
                    PILImage.BILINEAR
                )
                img = np.array(img_pil)

            # SAM 预处理
            from segment_anything.utils.transforms import ResizeLongestSide

            transform = ResizeLongestSide(sam.image_encoder.img_size)
            input_image = transform.apply_image(img)
            input_image_torch = torch.as_tensor(input_image, device=self.config.device).permute(2, 0, 1).contiguous()

            # 提取特征
            with torch.no_grad():
                # 获取图像嵌入
                features = sam.image_encoder(input_image_torch.unsqueeze(0))  # [1, C, H, W]

            # 转换为 [H, W, C] 格式
            features = features.squeeze(0).permute(1, 2, 0).cpu()  # [H, W, C]

            # 保存特征
            torch.save(features, feature_path)
            self.semantic_features[str(img_path)] = features
