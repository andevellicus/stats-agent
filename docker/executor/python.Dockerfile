# ====================
# STAGE 1: Build Stage
# ====================
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    gcc \
    gfortran \
    libreadline-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries (this stage has compilers)
RUN pip install --no-cache-dir --user \
    # Core Data Science
    pandas numpy matplotlib scikit-learn seaborn statsmodels scipy openpyxl \
    # R Integration
    rpy2 tzlocal \
    # Advanced ML
    xgboost lightgbm pymc \
    # Interactive Visualizations
    plotly \
    # Automated EDA
    ydata-profiling \
    shap pycox \
    lifelines arch \
    imblearn umap \
    pmdarima tbats prophet

# ====================
# STAGE 2: Runtime Stage
# ====================
FROM python:3.11-slim

# Install only runtime dependencies (NO compilers)
RUN apt-get update && apt-get install -y \
    r-base \
    libreadline8 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for running the executor
# Use UID 1000 to match typical host user (allows seamless file sharing via volumes)
# This ensures files created in mounted volumes have correct ownership
RUN useradd -m -u 1000 -s /bin/bash executor

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/executor/.local

# Copy executor script
COPY executor.py .

# Create workspace directory with proper permissions
RUN mkdir -p /app/workspaces && chown -R executor:executor /app

# Switch to non-root user
USER executor

# Add user packages to PATH
ENV PATH=/home/executor/.local/bin:$PATH

# Expose the port
EXPOSE 9999

# Run as non-root
CMD ["python", "-u", "executor.py"]
