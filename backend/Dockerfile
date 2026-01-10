FROM python:3.9-slim

WORKDIR /app

COPY .dvc ./.dvc
RUN apt-get update && apt-get install -y build-essential

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
RUN chmod +x start.sh
EXPOSE 8000
ENTRYPOINT ["./start.sh"]
