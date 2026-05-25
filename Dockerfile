FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 -s /bin/bash appuser
WORKDIR /app
COPY app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ .
RUN mkdir -p /data /model /results && chown -R appuser:appuser /app /data /model /results
USER appuser
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 MODEL_DIR=/model DATA_DIR=/data RESULTS_DIR=/results
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 CMD curl -f http://localhost:8080/health || exit 1
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
