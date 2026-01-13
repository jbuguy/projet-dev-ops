#!/usr/bin/env bash

set -e  # Exit immediately if a command fails

echo "Pulling data with DVC..."
dvc pull

echo "Building and starting containers..."
docker compose up --build -d

echo "Waiting for services to be ready..."
sleep 10

echo "Running ingestion..."
docker compose exec backend python -m app.ingest

echo "Restarting backend..."
docker compose restart backend

echo "Done. Now you can access the application at http://localhost:3000"
