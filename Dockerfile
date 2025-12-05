# Multi-stage build to include a Scaphandre binary and run the integrated lab

# Stage 1: build scaphandre from source using Rust
FROM rust:1.84 AS scaphandre-builder

# Install scaphandre via Cargo (binary will be in /usr/local/cargo/bin/scaphandre)
RUN cargo install scaphandre

# Stage 2: Python app image with scaphandre baked in
FROM python:3.11-slim AS app

WORKDIR /app

# System dependencies (curl/ca-certificates for HTTPS, optional)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Copy scaphandre binary from builder stage
COPY --from=scaphandre-builder /usr/local/cargo/bin/scaphandre /usr/local/bin/scaphandre

# Copy dependency manifests first for better Docker layer caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy the rest of the application code
COPY . .

# Environment for Scaphandre autostart inside the container
ENV SCAPHANDRE_CMD="scaphandre prometheus --bind :8080" \
    SCAPHANDRE_DEFAULT_URL="http://127.0.0.1:8080/metrics" \
    SCAPHANDRE_AUTOSTART=1 \
    PYTHONUNBUFFERED=1

# Expose the integrated lab port
EXPOSE 8001

# Default command: run the integrated lab
CMD ["uvicorn", "app_llm_behaviour_lab:app", "--host", "0.0.0.0", "--port", "8001"]
