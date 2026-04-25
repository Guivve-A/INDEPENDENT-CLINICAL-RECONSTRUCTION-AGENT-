# Day 1 — Agent 1 build. Final unified Dockerfile (with Agent 2 + 3) lands Day 5.
FROM rocm/primus:v26.2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "tests/test_stream_local.py"]
