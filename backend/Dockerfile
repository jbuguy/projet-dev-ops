FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x start.sh
EXPOSE 8000

ENTRYPOINT ["./start.sh"]
