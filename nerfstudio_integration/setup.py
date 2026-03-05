#!/usr/bin/env python
"""
Setup script for feature-3dgs nerfstudio integration.

This script helps set up the integration by:
1. Checking dependencies
2. Copying files to nerfstudio
3. Registering the method

Usage:
    python setup.py [--nerfstudio-path PATH]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")

    dependencies = {
        "nerfstudio": "pip install nerfstudio",
        "gsplat": "pip install gsplat>=1.0.0",
        "torch": "pip install torch",
    }

    missing = []
    for package, install_cmd in dependencies.items():
        try:
            __import__(package)
            print(f"  [green]✓[/green] {package}")
        except ImportError:
            print(f"  [red]✗[/red] {package} [missing]")
            missing.append((package, install_cmd))

    if missing:
        print("\nMissing dependencies:")
        for package, install_cmd in missing:
            print(f"  {package}: {install_cmd}")
        return False
    return True


def run_registration(args):
    """Run the registration script."""
    print("\nRunning registration...")
    register_script = Path(__file__).parent.parent / "scripts" / "register_feature_3dgs.py"

    if not register_script.exists():
        print(f"[red]Error: Registration script not found at {register_script}[/red]")
        return False

    cmd = [sys.executable, str(register_script)]
    if args.nerfstudio_path:
        cmd.extend(["--nerfstudio-path", args.nerfstudio_path])

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_tests(args):
    """Run the integration test suite."""
    print("\nRunning integration tests...")
    test_script = Path(__file__).parent.parent / "scripts" / "test_integration.py"

    if not test_script.exists():
        print(f"[red]Error: Test script not found at {test_script}[/red]")
        return False

    result = subprocess.run([sys.executable, str(test_script)])
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Setup feature-3dgs nerfstudio integration")
    parser.add_argument(
        "--nerfstudio-path",
        type=str,
        default=None,
        help="Path to nerfstudio installation",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency check",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip integration tests",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("feature-3dgs Nerfstudio Integration Setup")
    print("=" * 60)

    # Check dependencies
    if not args.skip_deps:
        if not check_dependencies():
            print("\n[red]Please install missing dependencies first[/red]")
            return 1

    # Run registration
    if not run_registration(args):
        print("\n[red]Registration failed[/red]")
        return 1

    # Run tests
    if not args.skip_tests:
        if not run_tests(args):
            print("\n[yellow]Tests failed, but integration may still work[/yellow]")

    print("\n" + "=" * 60)
    print("[green]Setup complete![/green]")
    print("=" * 60)
    print("\nYou can now use feature-3dgs:")
    print("  ns-train feature-3dgs --data <dataset> --semantic-feature-dir <features>")
    print("  ns-train feature-3dgs-speedup --data <dataset> --semantic-feature-dir <features>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
