# Docker Support

## Quick Start

### Build Image

```bash
docker build -t duplicate-detector:latest .
```

### Run Container

```bash
# Basic usage
docker run -v $(pwd)/input:/input -v $(pwd)/output:/output duplicate-detector:latest \
    --pdf /input/paper.pdf --output /output

# With custom config
docker run -v $(pwd):/workspace duplicate-detector:latest \
    --config /workspace/config.yaml --pdf /workspace/paper.pdf --output /workspace/output
```

### Docker Compose

```bash
docker-compose up
```

## GPU Support

For GPU acceleration:

```bash
docker build -f Dockerfile.gpu -t duplicate-detector:gpu .
docker run --gpus all -v $(pwd)/input:/input -v $(pwd)/output:/output duplicate-detector:gpu \
    --pdf /input/paper.pdf --output /output
```

## Image Variants

- `duplicate-detector:latest` - CPU-only, optimized for size
- `duplicate-detector:gpu` - CUDA-enabled, requires NVIDIA GPU
- `duplicate-detector:dev` - Development image with dev dependencies

