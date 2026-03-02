FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install Poppler for pdf2image
RUN apt-get update && apt-get install -y poppler-utils

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install -r requirements.txt

# Run server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]