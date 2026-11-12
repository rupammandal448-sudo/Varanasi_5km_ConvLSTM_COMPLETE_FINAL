FROM python:3.11-slim
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY server.py fetch_imd.py ./
COPY web ./web
ENV PORT=8000
EXPOSE 8000
# keep a single worker: the IMD fetch loop lives inside the process
CMD gunicorn -w 1 --threads 4 -b 0.0.0.0:${PORT} server:app
