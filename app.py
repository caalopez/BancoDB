import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from PIL import Image
import pdfplumber
import plotly.express as px
from prophet import Prophet
from utilidades import *

# Cargar datos
df = pd.read_csv("/Data/bancoDB_limpieza_2.csv")
df = df.drop(columns=['Unnamed: 0'])

st.markdown("<h1 style='text-align: center;'>📊Analisis de datos Banco de Sangre\n 🏥Clinica Cardio VID </h1>", unsafe_allow_html=True)



def main():

    menu= ["Inicio","Exploración de Datos", "Análisis de Variables Temporales", "Visualización de Edades", 
           "Visualización de Género", "Tipo de diferido, donante, donación", "Grupos Sanguíneos y Antígeno RH", 
           "Lugar de Donación", "Modelos Predictivos"]

    eleccion = st.sidebar.selectbox("Selecciona una opción", menu)
    # # Inicio
    if eleccion == "Inicio":
        st.title("Inicio")

    # # Exploración de Datos        
    elif eleccion == "Exploración de Datos":
        st.title("Exploración de Datos")
        st.dataframe(df)
        st.write("##### Dimensión DF", df.shape)

    # # Análisis de Variables Temporales
    elif eleccion == "Análisis de Variables Temporales":
        st.title("Análisis de Variables Temporales")
        donaciones_por_mes = procesar_donaciones(df)
        graficar_donaciones(donaciones_por_mes)
        min_donacion, max_donacion = calcular_rango_fechas(df)
        st.markdown(f"**📅 Desde:** {min_donacion}")  
        st.markdown(f"**📅 Hasta:** {max_donacion}")
        fig = mes_histograma(df)
        st.plotly_chart(fig)

    # # Visualización de Edades
    elif eleccion == "Visualización de Edades":
        st.title("Visualización de Edades")
        
        c1,c2 = st.columns([30,10])
        with c1:
            fig = crear_histograma(df)
            st.plotly_chart(fig)
            
        with c2:
            mostrar_boxplot(df)

        min_edad, max_edad = calcular_min_max(df, 'EDAD')
        st.write(f"📅 **Menor edad:** {min_edad}")
        st.write(f"📅 **Mayor edad:** {max_edad}")
        detectar_outliers(df)

    # # Visualización de Género
    elif eleccion  == "Visualización de Género":
        st.title("Visualización de Género")
        conteo_genero(df)
        promedio_edad(df)
        genero_donante(df)
        genero_gruposanguineo(df)
        

    # # Diferidos, Donantes y Donaciones
    elif eleccion == "Tipo de diferido, donante, donación":
        st.title("Tipo de Donante y Donación")
        #grafico_diferido(df)
        c1, c2 = st.columns([1,1])
        with c1:
            grafico_donante(df)
        with c2:
            grafico_donacion(df)
        edad_por_tipo_donacion(df)

    # # Grupos Sanguíneos y Antígeno RH
    elif eleccion == "Grupos Sanguíneos y Antígeno RH":
        st.title("Grupos Sanguíneos y Antígeno RH")
        grupos_sanguineos(df)
        grafico_rh(df)
        gruposanguineoVSdonacion(df)

        c1,c2 = st.columns([1,1])
        with c1:
            mostrar_grafico_grupo_rh(df)
        with c2:
            mostrar_grafico_grupo_rh_genero(df)

    # # Lugar de Donación
    elif eleccion == "Lugar de Donación":
        st.title("Lugar de Donación")
        lugar_donacion(df)
        donante_lugar_donacion(df)
        grafico_grupo_sanguineo_por_lugar(df)
        edad_por_tipo_donante(df)

    elif eleccion == "Modelos Predictivos":
        st.title("Modelos Predictivos")
        predecir_donaciones(df)
        st.write("📊 **Modelo de Regresión Lineal**")

if __name__ == "__main__":
  main()
