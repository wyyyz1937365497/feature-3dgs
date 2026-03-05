#!/usr/bin/env python
"""
Script to patch nerfstudio's method_configs.py to register feature-3dgs.

This script modifies the nerfstudio installation to include the feature-3dgs
configurations, allowing you to use `ns-train feature-3dgs` directly.

Usage:
    python register_feature_3dgs.py [--nerfstudio-path PATH]
"""

import argparse
import shutil
import sys
from pathlib import Path


def find_nerfstudio_path():
    """Find the nerfstudio installation path."""
    import nerfstudio
    return Path(nerfstudio.__file__).parent


def backup_file(file_path: Path) -> Path:
    """Create a backup of a file."""
    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"Created backup: {backup_path}")
    return backup_path


def patch_method_configs(nerfstudio_path: Path, project_root: Path):
    """Patch nerfstudio's method_configs.py to include feature-3dgs."""
    method_configs_path = nerfstudio_path / "configs" / "method_configs.py"

    if not method_configs_path.exists():
        print(f"Error: method_configs.py not found at {method_configs_path}")
        return False

    # Backup the original file
    backup_file(method_configs_path)

    # Read the original file
    with open(method_configs_path, "r") as f:
        content = f.read()

    # Check if already patched
    if "feature-3dgs" in content:
        print("method_configs.py already contains feature-3dgs configurations")
        return True

    # Find the import section and add our imports
    import_marker = "from nerfstudio.models.splatfacto import SplatfactoModelConfig"
    if import_marker in content:
        imports_to_add = f"""
# feature-3dgs imports
sys.path.insert(0, "{project_root}")
from feature_3dgs_extension.models.feature_3dgs import Feature3DGSModelConfig
from feature_3dgs_extension.data.dataparsers.semantic_feature_dataparser import SemanticFeatureDataparserConfig
"""
        content = content.replace(import_marker, import_marker + imports_to_add)

    # Add descriptions
    descriptions_marker = '    "splatfacto-big": "Larger version of Splatfacto with higher quality.",'
    if descriptions_marker in content:
        descriptions_to_add = '''
    "feature-3dgs": "Splatfacto with semantic feature support and text-guided editing.",
    "feature-3dgs-speedup": "Feature-3dgs with CNN decoder for faster training.",'''
        content = content.replace(descriptions_marker, descriptions_marker + descriptions_to_add)

    # Find the end of splatfacto-big config and add our configs
    # We'll add them before the discover_methods call at the end
    discover_marker = "discover_methods()"
    if discover_marker in content:
        # Read the feature-3dgs config template
        config_template = f"""

# feature-3dgs configurations
method_configs["feature-3dgs"] = TrainerConfig(
    method_name="feature-3dgs",
    steps_per_eval_image=100,
    steps_per_eval_batch=0,
    steps_per_save=2000,
    steps_per_eval_all_images=1000,
    max_num_iterations=30000,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            cache_images_type="uint8",
        ),
        model=Feature3DGSModelConfig(
            semantic_feature_dim=512,
            use_semantic_features=True,
            semantic_loss_weight=1.0,
            use_speedup=False,
            enable_editing=True,
        ),
    ),
    optimizers={{
        "means": {{
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        }},
        "features_dc": {{
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        }},
        "features_rest": {{
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        }},
        "opacities": {{
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        }},
        "scales": {{
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        }},
        "quats": {{"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None}},
        "semantic_features": {{
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None,
        }},
        "camera_opt": {{
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        }},
    }},
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["feature-3dgs-speedup"] = TrainerConfig(
    method_name="feature-3dgs",
    steps_per_eval_image=100,
    steps_per_eval_batch=0,
    steps_per_save=2000,
    steps_per_eval_all_images=1000,
    max_num_iterations=30000,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            cache_images_type="uint8",
        ),
        model=Feature3DGSModelConfig(
            semantic_feature_dim=512,
            use_semantic_features=True,
            semantic_loss_weight=1.0,
            use_speedup=True,
            enable_editing=True,
        ),
    ),
    optimizers={{
        "means": {{
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        }},
        "features_dc": {{
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        }},
        "features_rest": {{
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        }},
        "opacities": {{
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        }},
        "scales": {{
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        }},
        "quats": {{"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None}},
        "semantic_features": {{
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None,
        }},
        "cnn_decoder": {{
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None,
        }},
        "camera_opt": {{
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        }},
    }},
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

"""
        content = content.replace(discover_marker, config_template + discover_marker)

    # Write the patched file
    with open(method_configs_path, "w") as f:
        f.write(content)

    print(f"Patched {method_configs_path}")
    return True


def copy_integration_files(nerfstudio_path: Path, project_root: Path):
    """Copy integration files to nerfstudio."""
    import shutil

    # Create directories
    (nerfstudio_path / "models").mkdir(exist_ok=True)
    (nerfstudio_path / "data" / "dataparsers").mkdir(parents=True, exist_ok=True)
    (nerfstudio_path / "data" / "datasets").mkdir(parents=True, exist_ok=True)

    # Copy files
    files_to_copy = [
        (project_root / "nerfstudio" / "models" / "feature_3dgs.py",
         nerfstudio_path / "models" / "feature_3dgs.py"),
        (project_root / "nerfstudio" / "data" / "dataparsers" / "semantic_feature_dataparser.py",
         nerfstudio_path / "data" / "dataparsers" / "semantic_feature_dataparser.py"),
        (project_root / "nerfstudio" / "data" / "datasets" / "semantic_feature_dataset.py",
         nerfstudio_path / "data" / "datasets" / "semantic_feature_dataset.py"),
    ]

    for src, dst in files_to_copy:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {src.name} to {dst}")
        else:
            print(f"Warning: {src} not found, skipping")


def main():
    parser = argparse.ArgumentParser(description="Register feature-3dgs with nerfstudio")
    parser.add_argument(
        "--nerfstudio-path",
        type=str,
        default=None,
        help="Path to nerfstudio installation (auto-detected if not specified)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Path to feature-3dgs project root (auto-detected if not specified)",
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="Only copy files without patching method_configs.py",
    )

    args = parser.parse_args()

    # Find paths
    if args.nerfstudio_path:
        nerfstudio_path = Path(args.nerfstudio_path)
    else:
        nerfstudio_path = find_nerfstudio_path()
        print(f"Detected nerfstudio at: {nerfstudio_path}")

    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = Path(__file__).parent.parent
        print(f"Detected project root at: {project_root}")

    # Copy integration files
    print("\nCopying integration files...")
    copy_integration_files(nerfstudio_path, project_root)

    # Patch method_configs.py
    if not args.no_patch:
        print("\nPatching method_configs.py...")
        if patch_method_configs(nerfstudio_path, project_root):
            print("\n[green]Successfully registered feature-3dgs![/green]")
            print("\nYou can now use:")
            print("  ns-train feature-3dgs --data <dataset> --semantic-feature-dir <features>")
            print("  ns-train feature-3dgs-speedup --data <dataset> --semantic-feature-dir <features>")
        else:
            print("\n[red]Failed to patch method_configs.py[/red]")
            print("Please manually add the configurations.")
    else:
        print("\nSkipping method_configs.py patch (--no-patch flag set)")

    print("\nTo restore the original files, use:")
    print("  find nerfstudio -name '*.bak' -exec sh -c 'mv \"$1\" \"${1%.bak}\"' _ {} \\;")


if __name__ == "__main__":
    main()
