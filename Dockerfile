# Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    # Add X11 libraries needed for RDKit's Draw functionality
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching (external deps)
COPY requirements.txt .

# Install Python external dependencies
# Consider adding --no-cache-dir if image size is critical
RUN pip install -r requirements.txt

# Copy the application code (including src/, pyproject.toml, api/, etc.)
COPY . .

# --- Add this line to install your 'solpred' package ---
# This uses pyproject.toml to find and install the package from src/solpred
RUN pip install .

# Create required directories (if still needed, e.g., for model outputs/logs within container)
# If models are read-only and copied in, this might not be needed.
# RUN mkdir -p data/models # Adjust as needed

# Expose the port on which the application will run
EXPOSE 8000

# Set environment variables (PYTHONPATH should NO LONGER be needed)
# ENV PYTHONPATH=/app # REMOVE OR COMMENT OUT

# Command to run the application
CMD ["python", "-m", "api.main", "--host", "0.0.0.0", "--port", "8000"]