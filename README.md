# Persona‑Driven PDF Intelligence

**Round 1B - Adobe India "Connecting the Dots" Hackathon**

This project automatically finds and summarizes the most relevant parts of a set of PDFs for a given "persona" and "job_to_be_done." It takes a folder of PDFs along with a `query.json` file (containing "persona" and "job_to_be_done" fields) and outputs a `results.json` file containing:

- **Top K sections** ranked by semantic relevance (with keyword boosting)
- **A single, concise extractive summary** of those sections (de‑duplicated)

## Repository Structure

```
/
├── Dockerfile
├── README.md
├── approach_explanation.md
├── requirements.txt
├── main.py
├── extractive.py
├── embedder.py
├── pdf_utils.py
├── ranker.py
├── configs.py
├── cache_models.py
└── examples/
    ├── test_input/
    └── test_output/
```

## Dependencies

This project requires **Python 3.9** and the following packages:

- `sentence-transformers` - For semantic embeddings
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch framework
- `tqdm` - Progress bars
- `numpy` - Numerical computations
- `PyPDF2` or `pdfplumber` - PDF processing
- `scikit-learn` - Machine learning utilities
- `nltk` - Natural language processing

See `requirements.txt` for the complete list with version specifications.

## Model Caching & Offline Mode

To ensure the solution works in offline environments:

- We pre-download the `all-MiniLM-L6-v2` model into `/app/models` using `cache_models.py`
- The Dockerfile sets environment variables `ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to prevent internet access at runtime
- All required models and dependencies are cached during the Docker build process

## Build Instructions

Build the Docker image with the following command:

```bash
docker build --platform linux/amd64 -t adobe-hack-1b .
```

## Run Instructions (Offline)

Run the containerized solution in offline mode:

```bash
docker run --rm \
  --network none \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  adobe-hack-1b
```

**Note:** The `--network none` flag ensures the container runs completely offline.

## Input / Output

### Input
- **Location:** `/app/input/`
- **Contents:** 
  - Multiple PDF files to be analyzed
  - `query.json` file containing:
    ```json
    {
      "persona": "Description of the target persona",
      "job_to_be_done": "Specific task or objective"
    }
    ```

### Output
- **Location:** `/app/output/results.json`
- **Structure:**
  ```json
  {
    "metadata": {
      "query": {...},
      "processing_time": "...",
      "total_pdfs_processed": "...",
      "model_used": "..."
    },
    "extracted_sections": [
      {
        "pdf_name": "...",
        "section_title": "...",
        "content": "...",
        "relevance_score": "...",
        "page_number": "..."
      }
    ],
    "subsection_analysis": {
      "summary": "Concise extractive summary of top sections",
      "key_insights": [...],
      "confidence_score": "..."
    }
  }
  ```

## Configuration

Key configuration parameters can be found in `configs.py`:

- **TOP_K:** Number of top sections to extract (default: 10)
- **EMBEDDING_MODEL:** Model used for semantic similarity (`all-MiniLM-L6-v2`)
- **SUMMARY_MAX_LENGTH:** Maximum length for extractive summary
- **KEYWORD_BOOST_FACTOR:** Multiplier for keyword matching scores
- **SIMILARITY_THRESHOLD:** Minimum similarity score for section inclusion

You can modify these settings to fine-tune the extraction and summarization behavior.

## Approach Overview

For detailed information about the methodology, algorithms, and design decisions, please refer to `approach_explanation.md`.

## Example Usage

1. Place your PDF files in the `input/` directory
2. Create a `query.json` file with your persona and job-to-be-done
3. Build and run the Docker container
4. Check `output/results.json` for the processed results

See the `examples/` directory for sample inputs and expected outputs.

## Contact & License

**Developer:** [Your Name]  
**Email:** [your.email@example.com]  
**Event:** Adobe India "Connecting the Dots" Hackathon - Round 1B

This project is developed for educational and competition purposes. Please refer to the Adobe Hackathon terms and conditions for usage rights.

---

*Built with ❤️ for Adobe India Hackathon 2025*