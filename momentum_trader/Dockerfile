FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (exclude backtest, scripts, data, venv)
COPY shared/ shared/
COPY strategy/ strategy/
COPY strategy_2/ strategy_2/
COPY strategy_3/ strategy_3/
COPY config/ config/
COPY instrumentation/src/ instrumentation/src/
COPY instrumentation/config/ instrumentation/config/
COPY instrumentation/__init__.py instrumentation/__init__.py

# Create instrumentation data directories
RUN mkdir -p instrumentation/data/snapshots \
             instrumentation/data/trades \
             instrumentation/data/missed \
             instrumentation/data/daily \
             instrumentation/data/errors \
             instrumentation/data/scores \
             instrumentation/data/.sidecar_buffer

# Default command (overridden by docker-compose per service)
CMD ["python", "-m", "strategy"]
