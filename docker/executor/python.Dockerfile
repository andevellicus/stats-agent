# Use a more modern, slim Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install R and the necessary build dependencies for rpy2
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    gcc \
    gfortran \
    libreadline-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the executor script into the container
COPY executor.py .

# Install the comprehensive set of Python libraries
RUN pip install \
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
    shap \
    lifelines arch \
    imblearn umap \
    pmdarima tbats prophet 

# Expose the port the server will listen on
EXPOSE 9999

# The command to run when the container starts
CMD ["python", "-u", "executor.py"]
