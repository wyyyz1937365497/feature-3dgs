#!/usr/bin/env python
"""
Test script to verify the feature-3dgs integration with nerfstudio.

This script performs basic checks to ensure all components are properly set up.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    try:
        from feature_3dgs_extension.models.feature_3dgs import (
            Feature3DGSModel,
            Feature3DGSModelConfig,
            CNNDecoder,
        )
        print("  [green]✓[/green] Models imported successfully")

        from feature_3dgs_extension.data.dataparsers.semantic_feature_dataparser import (
            SemanticFeatureDataparser,
            SemanticFeatureDataparserConfig,
        )
        print("  [green]✓[/green] Dataparser imported successfully")

        from feature_3dgs_extension.data.datasets.semantic_feature_dataset import (
            SemanticFeatureDataset,
        )
        print("  [green]✓[/green] Dataset imported successfully")

        return True
    except ImportError as e:
        print(f"  [red]✗[/red] Import failed: {e}")
        return False


def test_model_config():
    """Test model configuration."""
    print("\nTesting model configuration...")

    try:
        from feature_3dgs_extension.models.feature_3dgs import Feature3DGSModelConfig

        config = Feature3DGSModelConfig(
            semantic_feature_dim=512,
            use_semantic_features=True,
            use_speedup=False,
            enable_editing=True,
        )

        print(f"  [green]✓[/green] Config created: {config._target}")
        print(f"    - semantic_feature_dim: {config.semantic_feature_dim}")
        print(f"    - use_semantic_features: {config.use_semantic_features}")
        print(f"    - use_speedup: {config.use_speedup}")
        print(f"    - enable_editing: {config.enable_editing}")

        return True
    except Exception as e:
        print(f"  [red]✗[/red] Config test failed: {e}")
        return False


def test_cnn_decoder():
    """Test CNN decoder."""
    print("\nTesting CNN decoder...")

    try:
        import torch
        from feature_3dgs_extension.models.feature_3dgs import CNNDecoder

        decoder = CNNDecoder(input_dim=128, output_dim=512)
        print("  [green]✓[/green] CNNDecoder created")

        # Test forward pass
        x = torch.randn(1, 128, 64, 64)
        output = decoder(x)
        print(f"  [green]✓[/green] Forward pass: {x.shape} -> {output.shape}")

        return True
    except Exception as e:
        print(f"  [red]✗[/red] CNN decoder test failed: {e}")
        return False


def test_dataparser_config():
    """Test dataparser configuration."""
    print("\nTesting dataparser configuration...")

    try:
        from feature_3dgs_extension.data.dataparsers.semantic_feature_dataparser import (
            SemanticFeatureDataparserConfig,
        )

        config = SemanticFeatureDataparserConfig(
            semantic_feature_dir="/path/to/features",
            semantic_feature_dim=512,
            use_speedup=False,
        )

        print(f"  [green]✓[/green] DataparserConfig created")
        print(f"    - semantic_feature_dir: {config.semantic_feature_dir}")
        print(f"    - semantic_feature_dim: {config.semantic_feature_dim}")

        return True
    except Exception as e:
        print(f"  [red]✗[/red] Dataparser config test failed: {e}")
        return False


def test_nerfstudio_integration():
    """Test nerfstudio is available."""
    print("\nTesting nerfstudio integration...")

    try:
        import nerfstudio
        print(f"  [green]✓[/green] Nerfstudio version: {nerfstudio.__version__}")

        from nerfstudio.models.splatfacto import SplatfactoModel
        print("  [green]✓[/green] SplatfactoModel available")

        from gsplat.rendering import rasterization
        print("  [green]✓[/green] gsplat rasterization available")

        return True
    except ImportError as e:
        print(f"  [red]✗[/red] Nerfstudio integration test failed: {e}")
        return False


def test_file_structure():
    """Test that all files are in place."""
    print("\nTesting file structure...")

    project_root = Path(__file__).parent.parent
    required_files = [
        "feature_3dgs_extension/models/feature_3dgs.py",
        "feature_3dgs_extension/data/dataparsers/semantic_feature_dataparser.py",
        "feature_3dgs_extension/data/datasets/semantic_feature_dataset.py",
        "feature_3dgs_extension/configs/feature_3dgs_configs.py",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  [green]✓[/green] {file_path}")
        else:
            print(f"  [red]✗[/red] {file_path} [missing]")
            all_exist = False

    return all_exist


def main():
    """Run all tests."""
    print("=" * 60)
    print("feature-3dgs Integration Test Suite")
    print("=" * 60)

    results = []

    results.append(("File Structure", test_file_structure()))
    results.append(("Imports", test_imports()))
    results.append(("Model Config", test_model_config()))
    results.append(("CNN Decoder", test_cnn_decoder()))
    results.append(("Dataparser Config", test_dataparser_config()))
    results.append(("Nerfstudio Integration", test_nerfstudio_integration()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[green]PASS[/green]" if result else "[red]FAIL[/red]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[bold green]All tests passed![/bold green]")
        print("\nYou can now use feature-3dgs with nerfstudio:")
        print("  ns-train feature-3dgs --data <dataset> --semantic-feature-dir <features>")
        return 0
    else:
        print("\n[bold red]Some tests failed. Please check the errors above.[/bold red]")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
