# devops project – university student chatbot

This project is a university chatbot developed following DevOps and MLOps best practices.  
It uses DVC for data and artifact versioning, Docker for reproducible environments, and GitHub Actions for CI/CD.

The application can be run in **development mode** (local embedding generation) or **production mode** (prebuilt Docker images with embeddings).

-----------

## Technologies

- Python 3.10
- FastAPI
- PyTorch
- Vector embeddings
- Docker & Docker Compose
- DVC (Google Drive remote)
- GitHub Actions
- React

---
## Prerequisites & API Keys

To enable the "Chat" capabilities (Generation), you need a **Hugging Face Access Token**.

1. **Get a Token:** Sign up at [Hugging Face](https://huggingface.co/settings/tokens) and generate a `read` token.
2. **Configuration:** You will pass this token as an environment variable (`HUGGINGFACEHUB_API_TOKEN`) when running the app.
## Repository Setup

```bash
git clone https://github.com/jbuguy/projet-dev-ops.git
cd projet-dev-ops
```
## Data and Artifacts (DVC)
Project data and ML artifacts are tracked using DVC.
```bash
pip install dvc[gdrive]
dvc pull
```
Raw documents used to build embeddings are located in `/data/raw`
## Development Mode
Development mode allows local generation of vector embeddings from raw documents
## configure secrets
Create a `.env` file in the `backend/` directory:
```toml
HUGGINGFACEHUB_API_TOKEN=hf_YourActualTokenHere
DATA_PATH=/app/backend/data
```
### build the app
```bash
docker compose up --build -d
```
### generate data embedding
```bash
docker compose exec backend python app/ingest.py
```
### access the application
open browser [http://localhost:3000](http://localhost:3000)
## production mode
you can use the prebuilt docker images that already include the embedding model and vector index
### images
| service | image name |
|---------|------------|
|Backend | wassimnefzi/university-chatbot-backend:latest|
|frontend| wassimnefzi/university-chatbot-frontend:latest|
1. create a file named `docker-compose.prod.yml`:
   ```yaml
   version: '3.8'
   services:
     backend:
       image: jbuguy/university-chatbot-backend:latest
       ports:
         - "8000:8000"
       environment:
         - HUGGINGFACEHUB_API_TOKEN=hf_YourTokenHere
     frontend:
       image: jbuguy/university-chatbot-frontend:latest
       ports:
         - "3000:5173"
       depends_on:
         - backend
    ```
2. run the application:
```bash
docker compose -f docker compose.prod.yml up -d
```
3. Open http://localhost:3000
## test
```bash
docker compose exec backend pytest
```
