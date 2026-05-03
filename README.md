# IntentClassiferOU-BE

Backend-only repository for the OU intent classification API.

## 1) Project structure

- app/main.py: FastAPI app entrypoint
- src/nlp/preprocessor.py: Vietnamese text preprocessing
- models/: place trained model files here
- requirements.txt: Python dependencies
- render.yaml: Render blueprint (optional)

## 2) Prerequisites

- Python 3.10+
- A trained model file: models/intent_classifier.pkl
- Label file: models/labels.json

## 3) Local run

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Health check:

```bash
curl http://localhost:8080/api/health
```

## 4) Security-related environment variables

- APP_ENV=production
- API_KEY=<strong random key>
- ALLOWED_ORIGINS=https://your-frontend-domain.com
- ALLOWED_HOSTS=your-service.onrender.com
- MAX_INPUT_CHARS=1200

If API_KEY is set, clients must send header: X-API-Key: <API_KEY>

## 5) Deploy on Render

Use the same settings from render.yaml or configure manually:

- Build Command: pip install -r requirements.txt
- Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
- Health Check Path: /api/health

After deployment, verify:

1. /api/health returns status ok.
2. /api/predict requires API key when API_KEY is configured.
3. CORS only allows configured frontend origins.
