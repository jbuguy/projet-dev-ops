#!/bin/bash

echo "Starting API..."
python -u -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
