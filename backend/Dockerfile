# JSJ Intelligence API in a container.
# Build context is the PROJECT ROOT (see docker-compose.yml), because the public demo library lives in demo/.
# The private library (data/) is never copied in: this image only ever contains public NIH pages.
FROM python:3.11-slim

WORKDIR /srv

# CPU-only PyTorch first: much smaller than the default build (no GPU libraries).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the other libraries before copying the code, so Docker can reuse this layer when only code changes.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/app backend/app
COPY demo demo

# The embedding model is downloaded once into /models (a Docker volume).
ENV HF_HOME=/models
ENV PYTHONUNBUFFERED=1

WORKDIR /srv/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
