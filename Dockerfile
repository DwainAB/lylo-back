FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Default: run the FastAPI server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
