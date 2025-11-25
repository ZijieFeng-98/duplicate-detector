"""
FastAPI REST API Server

Run the REST API server for programmatic access to duplicate detection.
"""

import uvicorn
from duplicate_detector.api.rest.api import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload for development
    )

