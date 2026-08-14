# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV HOME=/home/user
ENV PATH=$HOME/.local/bin:$PATH

# Install system dependencies for OpenCV and other ML libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a user to run the app (Hugging Face Spaces requires this)
RUN useradd -m -u 1000 user
USER user
WORKDIR $HOME/app

# Copy requirements first to leverage Docker cache
COPY --chown=user:user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user:user . .

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Command to run the Django server
CMD ["python", "neoledge_web/manage.py", "runserver", "0.0.0.0:7860"]
