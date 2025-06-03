import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Загрузка модели
model = joblib.load("model.pkl")

# Заголовок
st.title("🌸 Классификация цветка Iris")
st.write("Введите параметры цветка и получите предсказание его вида.")

# Ввод параметров пользователем
sepal_length = st.slider('Длина чашелистика (см)', 4.0, 8.0, 5.1)
sepal_width = st.slider('Ширина чашелистика (см)', 2.0, 4.5, 3.5)
petal_length = st.slider('Длина лепестка (см)', 1.0, 7.0, 1.4)
petal_width = st.slider('Ширина лепестка (см)', 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Отображение результата
species = ['Setosa', 'Versicolor', 'Virginica']
st.subheader("Результат предсказания:")
st.success(f"🌼 Предсказанный вид цветка: **{species[prediction[0]]}**")

st.subheader("Вероятности по классам:")
proba_df = pd.DataFrame(prediction_proba, columns=species)
st.dataframe(proba_df)
