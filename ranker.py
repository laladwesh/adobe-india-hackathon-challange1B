import numpy as np
from numpy.linalg import norm
from typing import List, Tuple

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between each row in A and each row in B:
    returns matrix of shape (len(A), len(B)).
    """
    # Normalize
    A_norm = A / np.clip(norm(A, axis=1, keepdims=True), 1e-10, None)
    B_norm = B / np.clip(norm(B, axis=1, keepdims=True), 1e-10, None)
    return np.dot(A_norm, B_norm.T)

def rank_sections(
    section_texts: List[str],
    persona_job: str,
    embedder,
    top_k: int
) -> List[Tuple[int, str, float]]:
    """
    Returns top_k sections ranked by similarity to the persona+job prompt.
    Each entry is (section_index, section_text, similarity_score).
    """
    section_embeds = embedder.encode(section_texts)
    prompt_embed = embedder.encode([persona_job])[0]  # single vector

    sims = (section_embeds @ prompt_embed) / (norm(section_embeds, axis=1) * norm(prompt_embed) + 1e-10)
    
    ranked_idx = np.argsort(-sims)[:top_k]
    return [(int(i), section_texts[i], float(sims[i])) for i in ranked_idx]
