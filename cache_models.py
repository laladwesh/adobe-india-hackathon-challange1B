# cache_models.py
# This script downloads the model and saves it to a specific local directory
# inside the Docker image, guaranteeing it is available for offline use.

from sentence_transformers import SentenceTransformer
import os

# --- Configuration ---
# Define the model we need from Hugging Face
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Define the local directory where the model will be saved
# This path is inside the Docker container.
SAVE_PATH = "/app/models/all-MiniLM-L6-v2"

# --- Caching Logic ---
print(f"Downloading model '{EMBEDDING_MODEL_ID}'...")

# Create a SentenceTransformer object from the model ID, which downloads it.
model = SentenceTransformer(EMBEDDING_MODEL_ID)

print(f"Saving model to local path: '{SAVE_PATH}'...")

# Save the downloaded model to our specified local path.
# This saves all necessary files (configs, weights, etc.) to that folder.
model.save(SAVE_PATH)

print("\nModel has been saved locally for offline use.")

