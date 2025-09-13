# slim + reproducible
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# system deps (optional but good for wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python deps first (layer caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy code + artifacts
COPY app.py ./app.py
COPY artifacts_rf ./artifacts_rf
COPY artifacts_scratch ./artifacts_scratch

# expose the dash port the app listens on
EXPOSE 8050

# health endpoint already exists in app.py (/health)
# use gunicorn to serve the Dash's WSGI server object
CMD ["tini", "--", "gunicorn", "-w", "2", "-b", "0.0.0.0:8050", "app:server"]