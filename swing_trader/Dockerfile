FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ shared/
COPY config/ config/
COPY strategy/ strategy/
COPY strategy_2/ strategy_2/
COPY strategy_3/ strategy_3/
COPY strategy_4/ strategy_4/
COPY instrumentation/ instrumentation/
COPY main_multi.py .
