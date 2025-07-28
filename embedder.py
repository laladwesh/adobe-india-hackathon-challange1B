from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    """
    A wrapper for the SentenceTransformer class that provides a consistent
    interface for encoding texts into high-quality embeddings. This version
    is highly optimized for speed on a CPU.
    """
    def __init__(self, model_name: str):
        """
        Initializes the Embedder with a specified Sentence Transformer model.
        The model is explicitly loaded onto the CPU as per the hackathon rules.
        """
        self.model = SentenceTransformer(model_name, device='cpu')
        print(f"Device set to use {self.model.device}")

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of texts into embeddings using the optimized model.
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
