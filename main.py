# main.py

import os
import json
import re
from datetime import datetime
import numpy as np
from tqdm import tqdm
import time # Import the time module

from configs import (
    TOP_K,
    EMBEDDING_MODEL,
)
from pdf_utils import load_sections
from embedder import Embedder
from sentence_transformers import util
# We no longer need the slow CrossEncoder or pipeline
# from sentence_transformers import CrossEncoder
# from transformers import pipeline

def extract_job_keywords(job: str):
    """
    Extracts a set of keywords from the job description string.
    """
    words = re.findall(r"\w+", job.lower())
    return [w for w in set(words) if len(w) > 3]

def summarize_extractive(text: str, prompt_embed: np.ndarray, embedder, num_sentences: int = 3) -> str:
    """
    A fast, extractive summarizer. It splits text into sentences, embeds them,
    and returns the top N sentences most similar to the user's prompt.
    """
    # Split text into sentences using a regular expression
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sents:
        return ""
    
    # Embed all sentences. This is fast with the optimized embedder.
    sent_embeds = embedder.encode(sents)
    
    # Calculate similarity of each sentence to the main prompt
    sims = util.cos_sim(prompt_embed, sent_embeds)[0].cpu().numpy()
    
    # Get the indices of the top N most similar sentences
    top_indices = np.argsort(-sims)[:num_sentences]
    
    # Return the top sentences in their original order
    return " ".join([sents[i] for i in sorted(top_indices)])

def main():
    # --- Start Timer ---
    start_time = time.time()

    import argparse
    parser = argparse.ArgumentParser("Round 1B: Persona-Driven PDF Intelligence")
    parser.add_argument("--input_dir",  required=True, help="Folder with PDFs + query.json")
    parser.add_argument("--output_dir", required=True, help="Where to write results.json")
    parser.add_argument("--query_file", default="query.json", help="Name of the spec file")
    args = parser.parse_args()

    # 1) Load query.json
    spec_path = os.path.join(args.input_dir, args.query_file)
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    persona = spec["persona"]["role"]
    job     = spec["job_to_be_done"]["task"]
    docs    = [d["filename"] for d in spec["documents"]]
    job_kws = extract_job_keywords(job)

    # 2) Extract all candidate sections
    all_secs = []
    for fn in docs:
        path = os.path.join(args.input_dir, fn)
        for sec in load_sections(path):
            all_secs.append({
                "document": fn, "page": sec["page"],
                "title": sec["section_title"] or "", "text": sec["text"] or ""
            })

    # 3) Fast bi-encoder embedding (relies on the optimized embedder.py)
    embedder     = Embedder(EMBEDDING_MODEL)
    texts        = [s["text"] for s in all_secs]
    prompt       = f"{persona.strip()} — {job.strip()}"
    
    # Encode prompt and texts together for efficiency
    all_embeddings = embedder.encode([prompt] + texts)
    prompt_embed, text_embeds = all_embeddings[0:1], all_embeddings[1:] # Keep prompt_embed as 2D array

    # 4) Rank sections using bi-encoder and keyword boosting
    sims = util.cos_sim(prompt_embed, text_embeds)[0]
    
    boosted_sims = []
    for i, sim_score in enumerate(sims):
        sec = all_secs[i]
        haystack = " ".join([sec["document"], sec["title"], sec["text"]]).lower()
        keyword_count = sum(1 for kw in job_kws if kw in haystack)
        boost = 1 + (0.1 * keyword_count)
        boosted_sims.append((i, float(sim_score) * boost))
    
    boosted_sims.sort(key=lambda x: -x[1])
    # The top indices are now our final ranking. No cross-encoder needed.
    top_idxs = [idx for idx, _ in boosted_sims[:TOP_K]]

    # 5) Build extracted_sections (No change here)
    extracted = []
    for rank, idx in enumerate(top_idxs, start=1):
        s = all_secs[int(idx)]
        extracted.append({
            "document": s["document"], "section_title": s["title"],
            "importance_rank": rank, "page_number": s["page"]
        })

    # 6) Summarize using the new, FAST extractive method
    subsection = []
    for idx in tqdm(top_idxs, desc="Summarizing sections"):
        s = all_secs[int(idx)]
        txt = s["text"]
        if not txt.strip():
            summary = "No content to summarize."
        else:
            # Call our new, fast summarizer
            summary = summarize_extractive(txt, prompt_embed, embedder)

        subsection.append({
            "document": s["document"], "refined_text": summary, "page_number": s["page"]
        })

    # 7) Assemble the final JSON output
    metadata = {
        "input_documents": docs, "persona": persona, "job_to_be_done": job,
        "processing_timestamp": datetime.utcnow().isoformat()
    }
    result = {
        "metadata": metadata, "extracted_sections": extracted, "subsection_analysis": subsection
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(result, out_f, indent=2, ensure_ascii=False)

    # --- End Timer and Print Duration ---
    end_time = time.time()
    duration = end_time - start_time
    print(f"✅ Wrote results to {os.path.join(args.output_dir, 'results.json')}")
    print(f"⏱️ Total execution time: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
