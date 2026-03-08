#!/usr/bin/env python
"""
Test script for LSeg optimization modules.

This script tests that the optimization modules can be imported and instantiated correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all optimization modules can be imported."""
    print("Testing imports...")

    try:
        from additional_utils.batch_processor import (
            BatchConfig,
            BatchFeatureExtractor,
            AsyncBatchProcessor,
        )
        print("✓ batch_processor imports successful")
    except Exception as e:
        print(f"✗ batch_processor import failed: {e}")
        return False

    try:
        from additional_utils.async_io import (
            AsyncIOScheduler,
            AsyncFeatureSaver,
            IOStats,
        )
        print("✓ async_io imports successful")
    except Exception as e:
        print(f"✗ async_io import failed: {e}")
        return False

    try:
        from additional_utils.data_preloader import (
            DataPreloader,
            BatchDataLoader,
            CachedImageLoader,
        )
        print("✓ data_preloader imports successful")
    except Exception as e:
        print(f"✗ data_preloader import failed: {e}")
        return False

    try:
        from additional_utils.encoding_models import (
            MultiEvalModule,
            vectorized_grid_inference,
        )
        print("✓ encoding_models imports successful")
    except Exception as e:
        print(f"✗ encoding_models import failed: {e}")
        return False

    return True


def test_batch_config():
    """Test BatchConfig creation."""
    print("\nTesting BatchConfig...")

    from additional_utils.batch_processor import BatchConfig

    config = BatchConfig(
        max_batch_size=4,
        prefetch_count=2,
        enable_mixed_precision=True,
    )

    assert config.max_batch_size == 4
    assert config.prefetch_count == 2
    assert config.enable_mixed_precision is True

    print("✓ BatchConfig creation successful")
    return True


def test_async_io():
    """Test AsyncIOScheduler creation."""
    print("\nTesting AsyncIOScheduler...")

    from additional_utils.async_io import AsyncIOScheduler

    scheduler = AsyncIOScheduler(max_workers=2, queue_size=4)

    assert scheduler.max_workers == 2
    assert scheduler.queue_size == 4

    stats = scheduler.get_stats()
    assert stats.successful_saves == 0
    assert stats.failed_saves == 0

    scheduler.shutdown()

    print("✓ AsyncIOScheduler creation successful")
    return True


def test_io_stats():
    """Test IOStats dataclass."""
    print("\nTesting IOStats...")

    from additional_utils.async_io import IOStats

    stats = IOStats(
        successful_saves=10,
        failed_saves=1,
        pending_count=5,
        total_bytes_saved=1024000,
    )

    assert stats.successful_saves == 10
    assert stats.failed_saves == 1
    assert stats.pending_count == 5
    assert stats.total_bytes_saved == 1024000

    print("✓ IOStats creation successful")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("LSeg Optimization Module Tests")
    print("=" * 60)

    tests = [
        test_imports,
        test_batch_config,
        test_async_io,
        test_io_stats,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
