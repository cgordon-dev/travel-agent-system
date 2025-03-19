FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# For monitoring and web server
RUN pip install --no-cache-dir prometheus-client opentelemetry-api opentelemetry-sdk opentelemetry-exporter-prometheus flask-cors

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data rl_data rl_models

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000
EXPOSE 8001

# Run the application
CMD ["python", "api_server.py"]