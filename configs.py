# How many top sections to extract + analyze
TOP_K = 6

# Use a smaller, faster embedding model to meet the 60-second time limit.
# This model is ~90MB vs the ~440MB of the previous one.
EMBEDDING_MODEL = "/app/models/all-MiniLM-L6-v2"

# Use a distilled version of BART for summarization to stay under the 1GB model limit.
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-6-6"
SUMMARIZATION_MAX_LEN = 200
SUMMARIZATION_MIN_LEN = 60
