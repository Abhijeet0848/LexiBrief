FROM python:3.10-slim

# System dependencies & AWS CLI
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app

# Install local package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 7860
EXPOSE 8080

CMD ["python", "app.py"]
