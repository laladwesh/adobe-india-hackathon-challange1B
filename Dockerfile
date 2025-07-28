# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add a non-root user for security best practices
RUN useradd -m myuser

# --- INSTALLATION (as root) ---
# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout=100 -r requirements.txt

# --- MODEL SAVING (as root) ---
# Create the target directory for our local models
RUN mkdir -p /app/models

# Copy the caching script
COPY cache_models.py .

# Run the script to download and SAVE the models to /app/models
RUN python cache_models.py

# --- FINAL SETUP ---
# Remove the temporary script now that the models are saved
RUN rm cache_models.py

# Copy the rest of your application code
COPY . .

# Change ownership of the entire /app directory to our non-root user
RUN chown -R myuser:myuser /app

# Switch to the non-root user for running the application
USER myuser

# Define the command to run your application
ENTRYPOINT ["python", "main.py", "--input_dir", "/app/input", "--output_dir", "/app/output"]
