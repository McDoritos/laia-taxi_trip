# ---- Base Python image ----
FROM python:3.13-slim AS base

# Avoid pyc & flush output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- Install OS deps ----
# Removed git since we aren't pulling inside the container anymore
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Install uv ----
RUN pip install uv

# ---- Set work directory ----
WORKDIR /app

# ---- Copy dependency files ----
COPY training/ .

# ---- Install dependencies ----
# Ensures reproducibility using uv
RUN uv sync --frozen

# ---- Entrypoint ----
ENTRYPOINT ["uv", "run", "python", "train_model.py"]