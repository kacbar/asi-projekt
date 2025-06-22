import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor

from azure.storage.blob import BlobServiceClient
import zipfile
import os
from autogluon.tabular import TabularPredictor
from dotenv import load_dotenv

if not os.path.exists("data/"):
    os.makedirs("data/")

def download_and_extract_model():
    load_dotenv()
    blob_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = "ml-models"
    blob_name = "06_models.zip"
    local_zip_path = "06_models.zip"
    extract_path = "data/06_models"

    blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
    blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)

    with open(local_zip_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        zip_ref.extractall("data/")

    # Wczytaj model z wypakowanego folderu
    predictor = TabularPredictor.load(extract_path)
    return predictor


# === Ścieżki ===
MODEL_PATH = "data/06_models"
DATA_PATH = "data/01_raw/student_depression_dataset.csv"

# === Wczytanie modelu i danych ===
predictor = download_and_extract_model()
raw_df = pd.read_csv(DATA_PATH)

# === Unikalne wartości dla dropdownów ===
degree_list = sorted(raw_df["Degree"].dropna().unique())
diet_list = sorted(raw_df["Dietary Habits"].dropna().unique())
city_list = sorted(raw_df["City"].dropna().unique())

# === UI ===
st.title("Predykcja depresji u studentów 🎓🧠")
st.markdown("Wprowadź dane studenta:")

# === Pola wejściowe ===
age = st.slider("Wiek", 16, 40, 20)
gender = st.selectbox("Płeć", [0, 1], format_func=lambda x: "Kobieta" if x == 0 else "Mężczyzna")
academic_pressure = st.slider("Presja akademicka", 0.0, 5.0, 2.5)
work_pressure = st.slider("Presja związana z pracą", 0.0, 5.0, 0.0)
financial_stress = st.slider("Stres finansowy", 0.0, 5.0, 2.5)
suicidal_thoughts = st.selectbox("Czy kiedykolwiek miałeś myśli samobójcze?", [0, 1], format_func=lambda x: "Nie" if x == 0 else "Tak")
family_history = st.selectbox("Czy w rodzinie występowały choroby psychiczne?", [0, 1], format_func=lambda x: "Nie" if x == 0 else "Tak")
study_satisfaction = st.slider("Zadowolenie z nauki", 0.0, 5.0, 2.5)
job_satisfaction = st.slider("Zadowolenie z pracy/nauki", 0.0, 5.0, 2.5)
cgpa = st.slider("Średnia ocen (CGPA)", 0.0, 10.0, 7.0)
work_study_hours = st.slider("Liczba godzin pracy/nauki dziennie", 0.0, 24.0, 6.0)
degree = st.selectbox("Stopień naukowy", degree_list)
diet = st.selectbox("Nawyki żywieniowe", diet_list)
city = st.selectbox("Miasto", city_list)

# === Dane wejściowe ===
input_df = pd.DataFrame([{
    "id": 999,
    "Age": age,
    "Gender": gender,
    "Academic Pressure": academic_pressure,
    "Work Pressure": work_pressure,
    "Financial Stress": financial_stress,
    "Have you ever had suicidal thoughts ?": suicidal_thoughts,
    "Family History of Mental Illness": family_history,
    "Study Satisfaction": study_satisfaction,
    "CGPA": cgpa,
    "Job Satisfaction": job_satisfaction,
    "Work/Study Hours": work_study_hours,
    "Degree": degree,
    "Dietary Habits": diet,
    "City": city,
    "Profession": "Student"
}])

# === Predykcja ===
if st.button("Sprawdź predykcję"):
    proba = predictor.predict_proba(input_df)
    st.write("📊 Prawdopodobieństwo depresji:", round(proba.loc[0, 1] * 100, 2), "%")
    prediction = predictor.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ Użytkownik może doświadczać depresji.")
    else:
        st.success("✅ Brak objawów depresji.")
