from sentence_transformers import SentenceTransformer
import os

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SAVE_PATH = "/app/models/all-MiniLM-L6-v2"

print(f"Downloading model '{EMBEDDING_MODEL_ID}'...")

model = SentenceTransformer(EMBEDDING_MODEL_ID)

print(f"Saving model to local path: '{SAVE_PATH}'...")

model.save(SAVE_PATH)

print("\nModel has been saved locally for offline use.")