FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY redditApp.py reddit_model_pipeline.joblib ./

CMD ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000"]