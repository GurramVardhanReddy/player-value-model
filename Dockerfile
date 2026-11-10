# bookworm: newer Debian releases no longer ship Java 17, which Spark 3.5 supports
FROM python:3.12-slim-bookworm

# Spark needs a Java runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless curl \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

RUN useradd --create-home app
WORKDIR /app
COPY pyproject.toml ./
COPY pvm ./pvm
RUN pip install --no-cache-dir .
COPY run.py ./
COPY scripts ./scripts
RUN chown -R app /app
USER app

# Mount data/raw (or let the script fetch it) and results/ as volumes
CMD ["bash", "-c", "scripts/fetch_data.sh data/raw && python run.py --data data/raw --out results"]
