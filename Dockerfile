FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 ollamadiffuser && \
    chown -R ollamadiffuser:ollamadiffuser /app
USER ollamadiffuser

# Expose ports
EXPOSE 8000 8001

# Default command
CMD ["ollamadiffuser", "--mode", "ui", "--host", "0.0.0.0", "--port", "8001"] 