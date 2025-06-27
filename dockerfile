# Używamy oficjalnego obrazu Pythona
FROM python:3.10-slim

# Ustawiamy katalog roboczy
WORKDIR /app

# Kopiujemy pliki projektu
COPY . .

# Instalujemy zależności
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Ustawiamy zmienne środowiskowe (mogą być też przekazywane przez --env-file podczas uruchomienia)
ENV AZURE_STORAGE_CONNECTION_STRING=${AZURE_STORAGE_CONNECTION_STRING}
ENV CONTAINER_NAME=${CONTAINER_NAME}
ENV BLOB_NAME=${BLOB_NAME}

# Otwieramy port Streamlit
EXPOSE 8501

COPY data/06_models/predictor.pkl /data/predictor.pkl

# Domyślne polecenie uruchamiające aplikację
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
