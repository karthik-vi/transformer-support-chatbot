# Deployment
Local: README quickstart. Docker: `docker build -t chatbot . && docker run --rm -p 8000:8000 chatbot` (TF-IDF fallback unless `artifacts/model` is mounted:
`-v $PWD/artifacts:/app/artifacts -e MODEL_DIR=/app/artifacts/model`). Inference ≈ 26 ms/query on CPU (batched). Model weights (~255 MB) are not committed.
Not production-ready: no auth, PII redaction, conversation state, monitoring of confidence drift, or content review of answers.
