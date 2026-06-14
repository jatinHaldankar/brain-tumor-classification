FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install required system dependencies (libgl1 is often needed for OpenCV, though we use PIL/TensorFlow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast Python package management
RUN pip install --no-cache-dir uv

# Copy the entire application first so the package source code is available
COPY . .

# Install dependencies and the package into the system python environment
# (We remove the -e flag because we don't need editable mode in production)
RUN uv pip install --system .

# Expose the port FastAPI runs on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
