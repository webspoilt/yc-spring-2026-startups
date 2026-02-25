FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY ai_engine/ ./ai_engine/

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Waiting for database..."\n\
while ! nc -z $DB_HOST $DB_PORT; do\n\
  sleep 1\n\
done\n\
echo "Database is ready!"\n\
\n\
exec uvicorn backend.main:app --host 0.0.0.0 --port $PORT\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Run the application
CMD ["/app/start.sh"]
