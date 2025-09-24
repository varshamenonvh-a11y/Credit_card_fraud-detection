# ==============================
# Dockerfile for Fraud Detection Flask App
# ==============================

# Use Python 3.13 base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files & buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (for numpy, pandas, scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (you can create requirements.txt separately)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into container
COPY . .

# Expose port (Render uses $PORT env variable automatically)
EXPOSE 5000

# Command to run Flask app with Gunicorn (production server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]