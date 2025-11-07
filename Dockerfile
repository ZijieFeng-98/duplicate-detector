FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package
COPY duplicate_detector/ ./duplicate_detector/
COPY pyproject.toml setup.py ./
RUN pip install --no-cache-dir -e .

# Copy main scripts
COPY ai_pdf_panel_duplicate_check_AUTO.py .
COPY streamlit_app.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
ENTRYPOINT ["python", "ai_pdf_panel_duplicate_check_AUTO.py"]

# Usage:
# docker build -t duplicate-detector .
# docker run -v $(pwd)/input:/input -v $(pwd)/output:/output duplicate-detector \
#     --pdf /input/paper.pdf --output /output

