# Approach Explanation: Persona-Driven PDF Intelligence

## Problem Statement

In today's information-rich environment, organizations struggle to extract relevant insights from large collections of PDF documents efficiently. Traditional document search relies on keyword matching, failing to understand semantic context and user intent. Our **Persona-Driven PDF Intelligence** solution addresses this by combining user personas with specific job-to-be-done contexts, enabling intelligent document summarization that delivers precisely what different stakeholders need from complex document repositories[1][44].

The "persona" + "job_to_be_done" approach ensures that a financial analyst seeking risk metrics receives different content extracts than a compliance officer reviewing the same documents for regulatory requirements.

## Section Extraction

Our system parses PDFs into semantically meaningful sections using document structure analysis techniques[22][25]. Rather than processing entire documents, we extract sections based on textual patterns, headers, and content boundaries. This section-level granularity is crucial because it preserves contextual coherence while enabling precise relevance scoring[28][31].

Section-level processing allows our system to identify and rank discrete chunks of information that maintain their original context, leading to more accurate semantic matching and better extractive summaries.

## Embedding & Ranking

We selected **all-MiniLM-L6-v2** as our bi-encoder model for its optimal balance of performance and efficiency[1][8]. This sentence transformer maps text to a 384-dimensional dense vector space, enabling fast semantic similarity calculations while maintaining strong accuracy for retrieval tasks[7][19].

Our system batch-embeds both the persona+job_to_be_done prompt and all PDF sections simultaneously, then computes cosine similarity scores[11]. We enhance ranking with lightweight keyword boosting, where exact term matches receive additional weight to complement semantic understanding[3][12].

## Top-K Selection

The system selects the highest-scoring K sections (configurable via `TOP_K` parameter) based on combined semantic similarity and keyword boost scores. Our keyword boost factor provides a practical balance between semantic understanding and exact term matching, ensuring both conceptually relevant and literally matching content surfaces appropriately[9].

## Combined Extractive Summarization

We evolved from per-section summaries to a unified extractive approach for improved coherence and efficiency[6]. Our `summarize_extractive` function splits the top-K sections into sentences, ranks each by prompt similarity, then applies a diversity filter to avoid redundancy by skipping sentences with >0.8 similarity to already selected content[3][6].

The final summary returns top sentences in their original order, preserving document flow while maximizing relevance and minimizing repetition.

## Performance & Constraints

Operating under 60-second execution limits with <1GB RAM constraints, we removed computationally expensive CrossEncoder re-ranking and abstractive summarization steps[6]. The all-MiniLM-L6-v2 model's efficiency (5x faster than larger alternatives while maintaining quality) proved essential for meeting these requirements[4][19].

## Offline Docker Setup

Our `cache_models.py` script pre-downloads the embedding model to `/app/models` during build time[1]. The Dockerfile sets `ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to ensure complete offline operation, critical for deployment in restricted environments[23][26].

## Key Design Decisions & Trade-offs

We prioritized extractive over abstractive summarization for reliability and speed, accepting slightly less natural language generation in favor of factual accuracy and performance[6][15]. Future improvements include lightweight abstractive fine-tuning, dynamic K selection based on document characteristics, and enhanced section parsing using advanced document structure recognition[22][40].

This approach delivers practical, persona-aware document intelligence within strict computational constraints while maintaining accuracy and offline deployment capability.