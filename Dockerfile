# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for GUI applications
RUN apt-get update && apt-get install -y \
    python3-tk \
    xvfb \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set display environment variable for GUI
ENV DISPLAY=:99

# Expose port if needed (for future web interface)
EXPOSE 8080

# Default command to run the application
CMD ["python", "main.py"]
