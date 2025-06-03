import streamlit as st
import pandas as pd
import joblib
from model_training import preprocess_text

st.title("📝 Классификация отзывов")
st.write("Загрузите CSV-файл с текстовыми отзывами для анализа их тональности.")

uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='cp1251')

    if 'text' not in df.columns:
        st.error("Файл должен содержать колонку 'text'")
    else:
        st.success("Файл успешно загружен!")
        st.write(df.head())

        model = joblib.load("model/sentiment_model.pkl")

        df['processed'] = df['text'].apply(preprocess_text)
        df['prediction'] = model.predict(df['processed'])
        df['prediction'] = df['prediction'].map({1: 'положительный', 0: 'отрицательный'})
        st.subheader("Результаты предсказания:")
        st.write(df[['text', 'prediction']])
