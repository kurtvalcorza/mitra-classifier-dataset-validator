# DIMER validator — Mitra tabular classification
# CPU-only: validation just parses CSVs. Keep it small and fast.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# validator.py is the tested implementation; validate.py is the portal-named
# entrypoint (DIMER's Pipeline Builder invokes `validate.py`) that delegates to it.
COPY validator.py validate.py ./

# DIMER launches this as a K8s Job with the env vars set.
CMD ["python", "validate.py"]
