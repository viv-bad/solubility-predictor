# Use Python 3.9 for better compatibility with ML libraries
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install core requirements first
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

# Copy and install ML requirements
COPY requirements-ml.txt .
# Add a longer timeout for ML dependencies
RUN pip install --no-cache-dir --timeout 300 -r requirements-ml.txt

# Copy the rest of the application
COPY . .

# Install the local package
RUN pip install .

# Create model directory if it doesn't exist
RUN mkdir -p models

# Expose the port 
EXPOSE ${PORT}

# Run the application with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]