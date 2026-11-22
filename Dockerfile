# Dockerfile — chạy demo Streamlit và toàn bộ pipeline phân tích.
# Build:  docker build -t bank-marketing .
# Chạy app: docker run -p 8501:8501 bank-marketing
# Chạy test: docker run --rm bank-marketing pytest -q
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app/ ./app/
COPY tests/ ./tests/
COPY data/ ./data/
COPY models/ ./models/
COPY README.md ./

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
