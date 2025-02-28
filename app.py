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
import os

# Cargar datos
#"/Users/carlos/Desktop/bancoDB/

ruta_csv = os.path.join("Data", "bancoDB_limpieza-2.csv")

if os.path.exists(ruta_csv):
    df = pd.read_csv(ruta_csv)
else:
    raise FileNotFoundError(f"El archivo {ruta_csv} no se encuentra. Verifica que esté en GitHub.")

#df = pd.read_csv("/workspaces/bancoDB/Data/bancoDB_limpieza-2.csv")
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

        # Ruta correcta si la imagen está en la carpeta "media"
        image_path = os.path.join("media", "clinica.jpg")

        # Verificar si la imagen existe antes de mostrarla
        if os.path.exists(image_path):
            st.image(image_path, caption="Clínica CardioVid", use_container_width=True)
        else:
            st.error(f"⚠️ Imagen no encontrada en: {image_path}. Verifica la ruta.")
        
        st.markdown("""  
### Introducción  

La **Clínica CardioVid** es una institución de referencia en atención cardiovascular, comprometida con la excelencia en el cuidado de la salud a través de un enfoque 
integral e innovador. Desde su fundación, ha trabajado para brindar servicios médicos de alta calidad, destacándose en la **prevención, diagnóstico y tratamiento** de 
enfermedades cardiovasculares. Su misión es mejorar la calidad de vida de los pacientes mediante un **servicio humanizado**, basado en la **tecnología de 
vanguardia** y el compromiso de un equipo altamente capacitado.  

Dentro de sus múltiples servicios, el **banco de sangre** desempeña un papel fundamental en la atención de pacientes que requieren 
**procedimientos quirúrgicos, tratamientos oncológicos y emergencias médicas**. La disponibilidad de **hemoderivados seguros y oportunos** 
depende de un sistema eficiente de **recolección, procesamiento y distribución** de unidades sanguíneas, lo que resalta la importancia de la **donación voluntaria**.  

Este proyecto tiene como objetivo **analizar los datos** relacionados con las **donaciones de sangre** en la Clínica CardioVid, 
con el fin de **identificar tendencias, mejorar la gestión del banco de sangre y promover estrategias** para aumentar la participación de donantes. 
A través del uso de herramientas de **análisis de datos**, se busca **optimizar la disponibilidad de sangre** y garantizar un abastecimiento adecuado 
para las necesidades hospitalarias.  
""")

    # # Exploración de Datos        
    elif eleccion == "Exploración de Datos":
        st.title("Exploración de Datos")
        #st.dataframe(df)
        #st.write("##### Dimensión DF", df.shape)
        explorar_datos(df)

    # # Análisis de Variables Temporales
    elif eleccion == "Análisis de Variables Temporales":
        st.title("Análisis de Variables Temporales")
        donaciones_por_mes = procesar_donaciones(df)
        graficar_donaciones(donaciones_por_mes)
        min_donacion, max_donacion = calcular_rango_fechas(df)
        st.markdown(f"**📅 Desde:** {min_donacion}")  
        st.markdown(f"**📅 Hasta:** {max_donacion}")
        distribucion_datos()
        fig = mes_histograma(df)
        st.plotly_chart(fig)
        distribucion2_datos()

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
        analizar_edad()

    # # Visualización de Género
    elif eleccion  == "Visualización de Género":
        st.title("Visualización de Género")
        conteo_genero(df)
        promedio_edad(df)
        genero_donante(df)
        genero_gruposanguineo(df)
        mostrar_conclusiones()
        

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
        donacion_analisis()

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
        gruposanguineo_conclusion()

    # # Lugar de Donación
    elif eleccion == "Lugar de Donación":
        st.title("Lugar de Donación")
        lugar_donacion(df)
        donante_lugar_donacion(df)
        lugar_conclusion()
        grafico_grupo_sanguineo_por_lugar(df)
        edad_por_tipo_donante(df)
        analisis_donantes()

    elif eleccion == "Modelos Predictivos":
        st.title("Modelos Predictivos")
        predecir_donaciones(df)
        st.write("📊 **Modelo de Regresión Lineal**")

        st.title("Explicación del Modelo de Predicción de Donaciones")

        st.header("Explicación del modelo de predicción")
        st.write(
            "El modelo se basa en datos históricos de donaciones para predecir tendencias futuras. "
            "Se usa Prophet porque es un modelo diseñado para capturar tendencias y estacionalidades "
            "en series temporales de manera automática."")

        st.header("Pasos de implementación en el código")

        st.subheader("Preprocesamiento de datos")
        st.write("- Se convierte la columna de fechas (`FECHA DONACION`) al formato `datetime`.")
        st.write("- Se agrupan las donaciones por mes. Prophet requiere que las columnas tengan nombres específicos:")
        st.write("  - `ds` → Fecha")
        st.write("  - `y` → Valor de la serie (cantidad de donaciones).")

        st.subheader("Entrenamiento del modelo")
        st.write("- Se crea una instancia de Prophet (`modelo = Prophet()`).")
        st.write("- Se entrena (`modelo.fit(donaciones_por_mes)`) con los datos históricos.")

        st.subheader("Generación de datos futuros")
        st.write("- Se define la fecha de inicio de predicción en enero de 2025.")
        st.write("- Se generan los meses a predecir hasta diciembre de 2027.")

        st.subheader("Predicción")
        st.write("- Se generan fechas futuras con `modelo.make_future_dataframe()`.")
        st.write("- Se predicen valores usando `modelo.predict(futuro)`.")

        st.subheader("Visualización en Streamlit")
        st.write("- Se usa Plotly para graficar la tendencia (`px.line()` con `yhat`, la predicción).")
        st.write("- Se muestra la gráfica en Streamlit con `st.plotly_chart(fig)`.")

        st.header("Resultado final")
        st.write(
            "El modelo genera una proyección de las donaciones desde enero de 2025 hasta diciembre de 2027, "
            "permitiendo analizar tendencias y posibles fluctuaciones en el futuro."




if __name__ == "__main__":
  main()
