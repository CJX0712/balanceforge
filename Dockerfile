# BalanceForge — CPU only, no GPU, no network needed at run time.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so the layer caches independently of the source.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY balanceforge ./balanceforge
COPY examples ./examples
COPY tests ./tests

RUN pip install --no-cache-dir --no-deps -e .

# Fail fast if the package cannot import.
RUN python -c "import balanceforge; print('balanceforge', balanceforge.__version__)"

ENTRYPOINT ["python", "examples/run_demo.py"]
