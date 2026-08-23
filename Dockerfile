# DIMER validator — Mitra tabular classification
# CPU-only: validation just parses CSVs. Keep it small and fast.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# validator.py owns task semantics; validate.py installs the repo-owned DIMER contract layer.
COPY validator.py validate.py contract_hardening.py ./

# DIMER launches this as a K8s Job with the env vars set.
CMD ["python", "validate.py"]
