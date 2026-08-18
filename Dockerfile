# DIMER validator — Mitra tabular classification
# CPU-only: validation just parses CSVs. Keep it small and fast.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY validator.py ./

# DIMER launches this as a K8s Job with the env vars set.
CMD ["python", "validator.py"]
