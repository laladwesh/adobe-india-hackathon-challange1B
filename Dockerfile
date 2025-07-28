FROM python:3.9-slim

WORKDIR /app

RUN useradd -m myuser

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout=100 -r requirements.txt

RUN mkdir -p /app/models

COPY cache_models.py .

RUN python cache_models.py

RUN rm cache_models.py

COPY . .

RUN chown -R myuser:myuser /app

USER myuser

ENTRYPOINT ["python", "main.py", "--input_dir", "/app/input", "--output_dir", "/app/output"]
