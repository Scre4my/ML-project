import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model("model.h5")
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

st.title("🖼️ Классификатор изображений (CIFAR-10)")
st.write("Загрузите изображение (32x32 или будет изменено автоматически)")

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    # Предобработка
    img = image.resize((32, 32))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Предсказание
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    st.subheader("🧠 Результат предсказания:")
    st.success(f"Объект на изображении: **{predicted_class}**")
