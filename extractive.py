import re
import numpy as np
from numpy.linalg import norm

def summarize_extractive(text: str, prompt_embed: np.ndarray, embedder, num_sentences: int = 3) -> str:
    """
    Split into sentences & pick the top‑k by cosine similarity to the prompt,
    with simple de‑duplication to avoid repeated sentences.
    """
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sents:
        return ""
    sent_embeds = embedder.encode(sents)
    sims = (sent_embeds @ prompt_embed) / (norm(sent_embeds, axis=1) * norm(prompt_embed) + 1e-10)

    selected_idxs = []
    seen_embeds = []
    threshold = 0.8
    for idx in np.argsort(-sims):
        if len(selected_idxs) >= num_sentences:
            break
        emb = sent_embeds[idx]
        if any((e @ emb) / (norm(e) * norm(emb) + 1e-10) > threshold for e in seen_embeds):
            continue
        selected_idxs.append(idx)
        seen_embeds.append(emb)

    if len(selected_idxs) < num_sentences:
        for idx in np.argsort(-sims):
            if idx not in selected_idxs:
                selected_idxs.append(idx)
            if len(selected_idxs) >= num_sentences:
                break

    selected_idxs = sorted(selected_idxs)
    return " ".join(sents[i] for i in selected_idxs)
