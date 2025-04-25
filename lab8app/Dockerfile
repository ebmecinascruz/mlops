FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary app files
COPY webapp.py .
COPY test_app.py .
COPY model_rf33.pkl .
COPY scaler.pkl .
COPY selected_feature_columns.pkl .
COPY numeric_columns.pkl .

EXPOSE 8000

CMD ["uvicorn", "webapp:app", "--host", "0.0.0.0", "--port", "8000"]