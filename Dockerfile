# Minimal, reproducible image for running the dqguard benchmark offline.
FROM python:3.12-slim

# Create an unprivileged user to run as.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Drop privileges.
USER appuser

# Default: run the benchmark but write nothing (keeps the container read-only).
ENTRYPOINT ["dqguard"]
CMD ["bench", "--no-write"]
