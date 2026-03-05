#!/usr/bin/env python
"""
Training script for feature-3dgs model.

This script provides a convenient interface for training the feature-3dgs model
with semantic features.

Usage:
    python train_feature_3dgs.py --data <path> --semantic-features <dir>
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "nerfstudio"))


def main():
    parser = argparse.ArgumentParser(description="Train feature-3dgs model")

    # Data arguments
    parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
    parser.add_argument(
        "--semantic-features",
        type=str,
        default=None,
        help="Path to directory containing pre-computed semantic features (.pt files)",
    )

    # Model arguments
    parser.add_argument(
        "--semantic-dim",
        type=int,
        default=512,
        help="Dimension of semantic features (default: 512)",
    )
    parser.add_argument(
        "--semantic-loss-weight",
        type=float,
        default=1.0,
        help="Weight for semantic feature loss (default: 1.0)",
    )
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="Use speedup mode with CNN decoder (faster training)",
    )
    parser.add_argument(
        "--no-editing",
        action="store_true",
        help="Disable text-guided editing functionality",
    )

    # Training arguments
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--iterations", type=int, default=30000, help="Number of training iterations")
    parser.add_argument("--eval-interval", type=int, default=1000, help="Evaluation interval")

    # Nerfstudio specific
    parser.add_argument(
        "--vis",
        type=str,
        default="viewer",
        choices=["viewer", "wandb", "tensorboard"],
        help="Visualization method",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()

    # Import nerfstudio components
    try:
        from nerfstudio.configs.base_config import ViewerConfig
        from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
        from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
        from nerfstudio.engine.optimizers import AdamOptimizerConfig
        from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
        from nerfstudio.engine.trainer import TrainerConfig
        from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
        from nerfstudio.models.feature_3dgs import Feature3DGSModelConfig
        from nerfstudio.utils.rich_utils import CONSOLE
    except ImportError as e:
        print(f"Error importing nerfstudio: {e}")
        print("Please ensure nerfstudio is installed: pip install nerfstudio")
        sys.exit(1)

    CONSOLE.print("[bold green]Starting feature-3dgs training[/bold green]")

    # Create trainer config
    config = TrainerConfig(
        method_name="feature-3dgs",
        output_dir=args.output,
        steps_per_eval_image=args.eval_interval,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=args.eval_interval,
        max_num_iterations=args.iterations,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    data=args.data,
                    load_3D_points=True,
                ),
                cache_images_type="uint8",
            ),
            model=Feature3DGSModelConfig(
                semantic_feature_dim=args.semantic_dim,
                use_semantic_features=args.semantic_features is not None,
                semantic_loss_weight=args.semantic_loss_weight,
                use_speedup=args.speedup,
                enable_editing=not args.no_editing,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=args.iterations,
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
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "semantic_features": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=args.iterations, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis=args.vis,
    )

    # Add semantic features path to metadata if provided
    if args.semantic_features:
        CONSOLE.print(f"[green]Loading semantic features from: {args.semantic_features}[/green]")
        # Store semantic feature directory for the dataset to use
        config.pipeline.datamanager.dataparser.semantic_feature_dir = args.semantic_features

    # Launch training using nerfstudio's entry point
    try:
        from nerfstudio.scripts.train import main as train_main

        # Simulate command line arguments for nerfstudio
        sys.argv = [
            "ns-train",
            "feature-3dgs",
            "--data",
            args.data,
            "--output-dir",
            args.output,
        ]

        if args.semantic_features:
            sys.argv.extend(["--semantic-feature-dir", args.semantic_features])
        if args.speedup:
            sys.argv.extend(["--speedup"])
        if args.no_editing:
            sys.argv.extend(["--no-editing"])

        train_main()

    except Exception as e:
        CONSOLE.print(f"[bold red]Error during training: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
