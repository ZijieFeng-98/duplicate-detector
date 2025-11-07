# Performance Optimization Guide

## Overview

This guide covers profiling, benchmarking, and optimizing the duplicate detection pipeline for better performance.

## Quick Profiling

### Using the Performance Utilities

```python
from duplicate_detector.utils.performance import (
    profile_context,
    memory_tracker,
    PerformanceBenchmark
)
from duplicate_detector import DuplicateDetector, DetectorConfig
from pathlib import Path

# Profile a single run
config = DetectorConfig.from_preset("balanced")
config.pdf_path = Path("paper.pdf")
config.output_dir = Path("results")

with profile_context(Path("profile.stats")):
    detector = DuplicateDetector(config=config)
    results = detector.analyze_pdf()

# View profile
import pstats
stats = pstats.Stats("profile.stats")
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Memory Tracking

```python
with memory_tracker() as mem:
    detector = DuplicateDetector(config=config)
    results = detector.analyze_pdf()

print(f"Peak memory: {mem['peak_mb']:.2f} MB")
```

## Benchmarking

### Run Benchmark Suite

```python
from duplicate_detector.utils.performance import PerformanceBenchmark

benchmark = PerformanceBenchmark(
    pdf_path=Path("paper.pdf"),
    output_dir=Path("benchmark_results")
)

# Compare presets
comparison = benchmark.compare_presets(["fast", "balanced", "thorough"])

# Single preset benchmark
result = benchmark.run_benchmark(
    preset="balanced",
    iterations=3,
    profile=True
)
```

### Command Line Benchmarking

```bash
python -m duplicate_detector.utils.performance paper.pdf balanced
```

## Optimization Strategies

### 1. Speed Optimization

```python
from duplicate_detector.utils.performance import optimize_config_for_speed

config = DetectorConfig.from_preset("balanced")
config = optimize_config_for_speed(config)

# Optimizations applied:
# - Lower DPI (100)
# - Higher CLIP threshold (0.97)
# - Disable ORB-RANSAC
# - Disable tier gating
# - Larger batch size (64)
```

### 2. Accuracy Optimization

```python
from duplicate_detector.utils.performance import optimize_config_for_accuracy

config = DetectorConfig.from_preset("balanced")
config = optimize_config_for_accuracy(config)

# Optimizations applied:
# - Higher DPI (200)
# - Lower CLIP threshold (0.94)
# - Enable ORB-RANSAC
# - Enable tier gating
# - Smaller batch size (16)
```

### 3. Memory Optimization

```python
config = DetectorConfig.from_preset("balanced")

# Reduce memory usage
config.performance.batch_size = 8  # Smaller batches
config.dpi = 100  # Lower resolution
config.feature_flags.enable_cache = True  # Enable caching
```

## Performance Tips

### 1. Enable Caching

Caching significantly speeds up repeated runs:

```python
config.feature_flags.enable_cache = True
config.performance.cache_version = "1.0"  # Increment to invalidate cache
```

### 2. Adjust Batch Size

- **Small batches (8-16):** Lower memory, slower
- **Medium batches (32):** Balanced (default)
- **Large batches (64+):** Faster, higher memory

### 3. Selective Feature Flags

Disable expensive features if not needed:

```python
# Fast mode
config.feature_flags.use_orb_ransac = False  # Skip geometric verification
config.feature_flags.use_tier_gating = False  # Skip tier classification

# Accurate mode
config.feature_flags.use_orb_ransac = True
config.feature_flags.use_tier_gating = True
```

### 4. DPI Selection

- **100 DPI:** Fast, good for quick checks
- **150 DPI:** Balanced (default)
- **200 DPI:** Slower, higher accuracy

### 5. Threshold Tuning

Higher thresholds = fewer comparisons = faster:

```python
config.duplicate_detection.sim_threshold = 0.97  # Higher = faster
config.duplicate_detection.phash_max_dist = 3  # Lower = faster
```

## Profiling Results Interpretation

### Common Bottlenecks

1. **CLIP Embeddings** - Usually 30-50% of runtime
   - Solution: Enable caching, increase batch size

2. **SSIM Computation** - Can be 20-30% of runtime
   - Solution: Increase threshold, disable for fast mode

3. **ORB-RANSAC** - Expensive but accurate
   - Solution: Disable for speed, enable for accuracy

4. **Panel Detection** - Usually fast (<10%)
   - Solution: Lower DPI if needed

### Profile Analysis

```python
import pstats

stats = pstats.Stats("profile.stats")

# Sort by cumulative time
stats.sort_stats('cumulative')
stats.print_stats(20)

# Sort by total time
stats.sort_stats('tottime')
stats.print_stats(20)
```

## Benchmark Results Format

```json
{
  "preset": "balanced",
  "iterations": 3,
  "times": [82.1, 81.5, 82.8],
  "avg_time": 82.13,
  "min_time": 81.5,
  "max_time": 82.8,
  "memory_peak_mb": 745.2,
  "panels": 107,
  "duplicates": 108
}
```

## Performance Baselines

### Tested on 34-page scientific paper:

**Fast Preset:**
- Runtime: ~2 minutes
- Memory: ~500 MB
- Panels: 107
- Duplicates: 108

**Balanced Preset (Recommended):**
- Runtime: ~5 minutes
- Memory: ~750 MB
- Panels: 107
- Duplicates: 108

**Thorough Preset:**
- Runtime: ~10 minutes
- Memory: ~1 GB
- Panels: 107
- Duplicates: 108

## GPU Acceleration

For faster CLIP processing:

```python
# GPU is automatically used if available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Further Optimization

1. **Parallel Processing:** Use multiprocessing for independent operations
2. **Memory Mapping:** Already implemented for cache loading
3. **Incremental Processing:** Process pages in batches
4. **Early Stopping:** Stop after finding enough duplicates

## Troubleshooting

**Out of Memory:**
- Reduce batch size
- Lower DPI
- Disable ORB-RANSAC
- Process in smaller chunks

**Too Slow:**
- Enable caching
- Use "fast" preset
- Increase thresholds
- Disable expensive features

**Low Accuracy:**
- Use "thorough" preset
- Lower thresholds
- Enable all features
- Increase DPI

