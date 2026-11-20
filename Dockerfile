# JSJ Intelligence: Streamlit chat page in a container.
# The image holds the code only. The copyrighted articles and the chunk files
# (the data/ folder) are NOT copied in; docker-compose.yml mounts them at run time.
FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch first: much smaller than the default build (no GPU libraries).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the other libraries before copying the code, so Docker can reuse this
# layer when only the code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app.py .

# The embedding model is downloaded once into /models (a Docker volume).
ENV HF_HOME=/models
ENV PYTHONUNBUFFERED=1

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
