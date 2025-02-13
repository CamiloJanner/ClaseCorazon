import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo y el escalador
modelo_knn = joblib.load('modelo_knn.bin')
escalador = joblib.load('escalador.bin')

# Título y descripción
st.title('Asistente IA para cardiólogos')
st.write("""
    Bienvenido al asistente IA diseñado para ayudar a los cardiólogos en la predicción de problemas cardíacos.
    Esta aplicación utiliza un modelo de inteligencia artificial basado en KNN (K-Nearest Neighbors) que predice
    si una persona tiene o no problemas cardíacos basado en la edad y el colesterol, utilizando un conjunto de datos
    previamente entrenado.
""")

# Pestañas: Ingreso de datos y predicción
tabs = st.tabs(["Ingreso de datos", "Predicción"])

# Ingreso de datos (tab 1)
with tabs[0]:
    st.header("Ingreso de Datos")
    st.write("""
        Ingrese los siguientes datos para hacer la predicción:
    """)
    edad = st.slider("Edad", 18, 80, 40)
    colesterol = st.slider("Colesterol", 50, 600, 200)

    st.write("Por favor, ajuste los valores en los deslizadores para introducir los datos de la persona.")

# Predicción (tab 2)
with tabs[1]:
    st.header("Predicción")
    
    # Si el botón de predecir es presionado
    if st.button('Predecir'):
        # Crear el dataframe con los datos de entrada y asegurarse de que las columnas son las correctas
        datos_entrada = pd.DataFrame([[edad, colesterol]], columns=['edad', 'colesterol'])

        # Normalizar los datos usando el escalador cargado
        datos_normalizados = escalador.transform(datos_entrada)

        # Hacer la predicción con el modelo KNN
        prediccion = modelo_knn.predict(datos_normalizados)

        # Mostrar la predicción
        if prediccion[0] == 1:
            st.write("¡Alerta! La persona **tiene un problema cardíaco**.")
            # Mostrar la imagen si tiene problema cardíaco
            st.image("https://www.meditip.lat/wp-content/uploads/2017/11/Enfermedad-del-coraz%C3%B3n.jpg", caption="Problema Cardíaco")
        else:
            st.write("¡Excelente! La persona **no tiene problemas cardíacos**.")
            # Mostrar la imagen si no tiene problema cardíaco
            st.image("https://static.vecteezy.com/system/resources/previews/011/646/410/non_2x/happy-african-american-black-man-with-thumbs-up-like-gesture-in-casual-bright-shirt-white-background-photo.jpg", caption="Sin Problema Cardíaco")
