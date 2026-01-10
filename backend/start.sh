#!/bin/bash

echo "pulling data"
dvc pull
echo "starting application..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
