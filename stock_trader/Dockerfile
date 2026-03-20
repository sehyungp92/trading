FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY strategy_iaric/ strategy_iaric/
COPY instrumentation/ instrumentation/
COPY shared/ shared/
COPY strategy_orb/ strategy_orb/

RUN mkdir -p \
    /app/data/strategy_iaric \
    /app/data/strategy_orb \
    /app/instrumentation/data/.sidecar_buffer \
    /app/instrumentation/data/daily \
    /app/instrumentation/data/errors \
    /app/instrumentation/data/filter_decisions \
    /app/instrumentation/data/heartbeats \
    /app/instrumentation/data/indicators \
    /app/instrumentation/data/missed \
    /app/instrumentation/data/orderbook \
    /app/instrumentation/data/orders \
    /app/instrumentation/data/scores \
    /app/instrumentation/data/snapshots \
    /app/instrumentation/data/trades

CMD ["python", "-m", "strategy_iaric"]
