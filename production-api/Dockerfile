FROM python:3.12-slim

WORKDIR /app

# 1. Install curl for your healthcheck to work (slim images don't have it)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 2. Fixed typo: changed appuser::appuser to appuser:appuser
RUN useradd --create-home appuser && chown appuser:appuser /app

RUN pip install uv

COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser uv.lock* .

USER appuser

RUN uv sync --frozen --no-dev

COPY --chown=appuser:appuser production-api/app/ app/

# 3. Removed the duplicate `RUN useradd` and `USER appuser` commands that were here

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 4. Fixed spaces inside the CMD strings (" -- host" -> "--host")
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]