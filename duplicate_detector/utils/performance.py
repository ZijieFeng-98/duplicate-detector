"""
Performance Profiling and Optimization Utilities

Tools for profiling and optimizing the duplicate detection pipeline.
"""

import cProfile
import pstats
import io
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import time
import tracemalloc
from contextlib import contextmanager

if TYPE_CHECKING:
    from duplicate_detector.api.detector import DuplicateDetector
    from duplicate_detector.models.config import DetectorConfig


@contextmanager
def profile_context(output_file: Optional[Path] = None):
    """
    Context manager for profiling code execution.
    
    Example:
        with profile_context(Path("profile.stats")):
            detector.analyze_pdf()
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        yield profiler
    finally:
        profiler.disable()
        
        if output_file:
            profiler.dump_stats(str(output_file))
            print(f"Profile saved to {output_file}")
        else:
            # Print to stdout
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)
            print(s.getvalue())


@contextmanager
def memory_tracker():
    """
    Context manager for tracking memory usage.
    
    Example:
        with memory_tracker() as tracker:
            detector.analyze_pdf()
        print(f"Peak memory: {tracker['peak'] / 1024 / 1024:.2f} MB")
    """
    tracemalloc.start()
    start_time = time.time()
    
    try:
        yield {}
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.time() - start_time
        
        result = {
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024,
            'elapsed_seconds': elapsed
        }
        
        print(f"Memory Usage:")
        print(f"  Current: {result['current_mb']:.2f} MB")
        print(f"  Peak: {result['peak_mb']:.2f} MB")
        print(f"  Time: {result['elapsed_seconds']:.2f} seconds")
        
        # Update the context dict
        if isinstance(result, dict):
            result.update(result)


class PerformanceBenchmark:
    """
    Benchmark suite for duplicate detection performance.
    """
    
    def __init__(self, pdf_path: Path, output_dir: Path):
        """
        Initialize benchmark.
        
        Args:
            pdf_path: Path to test PDF
            output_dir: Output directory for results
        """
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def run_benchmark(
        self,
        preset: str = "balanced",
        iterations: int = 3,
        profile: bool = False
    ) -> dict:
        """
        Run benchmark with given preset.
        
        Args:
            preset: Configuration preset
            iterations: Number of iterations
            profile: Whether to enable profiling
        
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Benchmark: {preset} preset")
        print(f"{'='*60}")
        
        from duplicate_detector.api.detector import DuplicateDetector
        from duplicate_detector.models.config import DetectorConfig

        times = []
        memory_peaks = []
        
        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}")
            
            # Create config
            config = DetectorConfig.from_preset(preset)
            config.pdf_path = self.pdf_path
            config.output_dir = self.output_dir / f"run_{i+1}"
            config.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run with profiling if requested
            if profile and i == 0:  # Profile only first iteration
                profile_file = self.output_dir / f"profile_{preset}.stats"
                with profile_context(profile_file):
                    with memory_tracker() as mem:
                        detector = DuplicateDetector(config=config)
                        start = time.time()
                        results = detector.analyze_pdf()
                        elapsed = time.time() - start
                        memory_peaks.append(mem.get('peak_mb', 0))
            else:
                with memory_tracker() as mem:
                    detector = DuplicateDetector(config=config)
                    start = time.time()
                    results = detector.analyze_pdf()
                    elapsed = time.time() - start
                    memory_peaks.append(mem.get('peak_mb', 0))
            
            times.append(elapsed)
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Panels: {results.metadata.get('num_panels', 0)}")
            print(f"  Duplicates: {results.total_pairs}")
            print(f"  Tier A: {len(results.tier_a_pairs)}")
            print(f"  Tier B: {len(results.tier_b_pairs)}")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_memory = sum(memory_peaks) / len(memory_peaks) if memory_peaks else 0
        
        result = {
            'preset': preset,
            'iterations': iterations,
            'times': times,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'memory_peak_mb': avg_memory,
            'panels': results.metadata.get('num_panels', 0),
            'duplicates': results.total_pairs
        }
        
        self.results.append(result)
        
        print(f"\nSummary:")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Min time: {min_time:.2f}s")
        print(f"  Max time: {max_time:.2f}s")
        print(f"  Peak memory: {avg_memory:.2f} MB")
        
        return result
    
    def compare_presets(self, presets: list = ["fast", "balanced", "thorough"]) -> dict:
        """
        Compare performance across presets.
        
        Args:
            presets: List of presets to compare
        
        Returns:
            Comparison results
        """
        print(f"\n{'='*60}")
        print("Comparing Presets")
        print(f"{'='*60}")
        
        comparison = {}
        
        for preset in presets:
            result = self.run_benchmark(preset=preset, iterations=1, profile=False)
            comparison[preset] = result
        
        # Print comparison table
        print(f"\n{'Preset':<15} {'Time (s)':<12} {'Memory (MB)':<15} {'Panels':<10} {'Duplicates':<12}")
        print("-" * 70)
        for preset, result in comparison.items():
            print(f"{preset:<15} {result['avg_time']:<12.2f} {result['memory_peak_mb']:<15.2f} "
                  f"{result['panels']:<10} {result['duplicates']:<12}")
        
        return comparison
    
    def save_results(self, output_file: Path):
        """Save benchmark results to JSON."""
        import json
        
        output_file = Path(output_file)
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")


def optimize_config_for_speed(config: "DetectorConfig") -> "DetectorConfig":
    """
    Optimize configuration for maximum speed.
    
    Returns:
        Optimized DetectorConfig
    """
    config.dpi = 100  # Lower DPI
    config.duplicate_detection.sim_threshold = 0.97  # Higher threshold (fewer comparisons)
    config.duplicate_detection.phash_max_dist = 3  # Stricter pHash
    config.feature_flags.use_phash_bundles = True  # Keep bundles (fast)
    config.feature_flags.use_orb_ransac = False  # Disable ORB (slow)
    config.feature_flags.use_tier_gating = False  # Skip tier classification
    config.performance.batch_size = 64  # Larger batches
    config.feature_flags.enable_cache = True  # Enable caching
    
    return config


def optimize_config_for_accuracy(config: "DetectorConfig") -> "DetectorConfig":
    """
    Optimize configuration for maximum accuracy.
    
    Returns:
        Optimized DetectorConfig
    """
    config.dpi = 200  # Higher DPI
    config.duplicate_detection.sim_threshold = 0.94  # Lower threshold (more comparisons)
    config.duplicate_detection.phash_max_dist = 5  # More lenient pHash
    config.feature_flags.use_phash_bundles = True
    config.feature_flags.use_orb_ransac = True  # Enable ORB
    config.feature_flags.use_tier_gating = True  # Enable tier classification
    config.performance.batch_size = 16  # Smaller batches (more accurate)
    config.feature_flags.enable_cache = True
    
    return config


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python performance.py <pdf_path> [preset]")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    preset = sys.argv[2] if len(sys.argv) > 2 else "balanced"
    
    output_dir = Path("benchmark_results")
    benchmark = PerformanceBenchmark(pdf_path, output_dir)
    
    result = benchmark.run_benchmark(preset=preset, iterations=3, profile=True)
    benchmark.save_results(output_dir / "benchmark_results.json")

