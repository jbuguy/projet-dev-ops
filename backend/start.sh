#!/bin/bash


# Define the path to the artifacts directory and the specific file
ARTIFACTS_DIR="data/artifacts"
INDEX_FILE="$ARTIFACTS_DIR/vector_index.faiss"

# Check if the artifact directory exists, create it if not
if [ ! -d "$ARTIFACTS_DIR" ]; then
  echo "Creating artifacts directory..."
  mkdir -p "$ARTIFACTS_DIR"
fi

# Check if the vector index exists
if [ ! -f "$INDEX_FILE" ]; then
  echo "Vector index not found at $INDEX_FILE."
  echo "Running ingestion script..."
  
  # Run the ingestion script
  python app/ingest.py
  
  # Check if ingestion succeeded
  if [ $? -eq 0 ]; then
    echo "Ingestion completed successfully."
    echo "---------------------------------"
  else
    echo "Ingestion failed. Exiting."
    exit 1
  fi
else
  echo "Vector index found. Skipping ingestion."
fi

# Start the application
echo "Starting API..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
