# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the executor script into the container
COPY executor.py .
COPY .env .

# Install any Python libraries you need
RUN pip install pandas numpy matplotlib scikit-learn seaborn statsmodels

# Expose the port the server will listen on
EXPOSE 9999

# The command to run when the container starts
CMD ["python", "-u", "executor.py"]