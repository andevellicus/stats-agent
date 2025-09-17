# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the executor script into the container
COPY executor.py .

# (Optional) Install any Python libraries you need for your stats agent
# For example, to install pandas and numpy:
RUN pip install pandas numpy matplotlib scikit-learn seaborn 

# Expose the port the server will listen on
EXPOSE 9999

# The command to run when the container starts
CMD ["python", "-u", "executor.py"]