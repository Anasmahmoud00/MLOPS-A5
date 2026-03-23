# Dockerfile
FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

# Simulate downloading the model from MLflow
RUN echo "Downloading model for RUN_ID=$RUN_ID"

# Default command
CMD ["echo", "Container ready for RUN_ID"]