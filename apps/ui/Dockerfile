FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir '.[ui]'

CMD ["python", "-u", "-m", "polyglot_ui.app"]
