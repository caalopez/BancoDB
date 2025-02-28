import pandas as pd
import streamlit as st
import utilidades as util
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



df = pd.read_csv("/Users/carlos/Desktop/bancoDB/Data/bancoDB_limpieza-2.csv")
df = df.drop(columns=['Unnamed: 0'])

# def generarMenu():
#     with st.sidebar:
#         col1, col2 = st.columns(2)
#         with col1:
#             image = Image.open("media/logo.png")
#             st.image(image, use_container_width=False)
#         with col2:
#             st.header('UnDato')

#         st.page_link("pages/inicio.py", label='Inicio')
#         st.page_link("pages/exploracion.py", label='Exploracion de datos')
#         st.page_link("pages/analisis_temporal.py", label='Analisis Temporal')





def calcular_rango_fechas(df):
    """Calcula los valores mínimos y máximos de las fechas en el DataFrame."""
    min_donacion = df['FECHA DONACION'].min()
    max_donacion = df['FECHA DONACION'].max()
    #min_nac = df['FECHA NACIMIENTO'].min()
    #max_nac = df['FECHA NACIMIENTO'].max()
    
    return min_donacion, max_donacion #min_nac, max_nac


def procesar_donaciones(df):
    """Convierte fechas y cuenta donaciones por mes y año."""
    df["FECHA DONACION"] = pd.to_datetime(df["FECHA DONACION"])
    df["AÑO"] = df["FECHA DONACION"].dt.year
    df["MES"] = df["FECHA DONACION"].dt.month

    # Agrupar por año y mes
    donaciones_por_mes = df.groupby(["AÑO", "MES"]).size().reset_index(name="CANTIDAD")

    # Crear columna de fecha para el gráfico
    donaciones_por_mes["FECHA"] = pd.to_datetime(donaciones_por_mes["AÑO"].astype(str) + "-" + donaciones_por_mes["MES"].astype(str))

    return donaciones_por_mes


def graficar_donaciones(donaciones_por_mes):
    """Genera un gráfico de tendencia de donaciones."""
    fig = px.line(
        donaciones_por_mes, x="FECHA", y="CANTIDAD", 
        markers=True, title="Tendencia de Donaciones en el Tiempo",
        labels={"FECHA": "Fecha", "CANTIDAD": "Número de Donaciones"},
        template="plotly_dark"
    )
    # Centrar el título del gráfico
    fig.update_layout(title_x=0.4)

    # Mostrar en Streamlit
    st.plotly_chart(fig)


def crear_histograma(df):
    fig = px.histogram(df, x="EDAD", nbins=50, marginal="box", text_auto = True)
    fig.update_layout(
        title={
            'text': "Distribución de Edades de los Donantes",
            'x': 0.5,  # Centrar el título
            'xanchor': 'center',
            'font': {'size': 24}  # Aumentar el tamaño del título
        },
        xaxis_title="Edad del Donante",
        yaxis_title="Cantidad de Donantes"
    )
    return fig

def mes_histograma(df):
    
    df.groupby("MES").size().reset_index(name="CANTIDAD")
    fig = px.histogram(df, x="MES", nbins=50, marginal="box", title="Distribución de Mes de los Donantes", text_auto=True)
    fig.update_layout(
        title={
            'text': "Distribución de Mes de donación ",
            'x': 0.5,  # Centrar el título
            'xanchor': 'center',
            'font': {'size': 24}  # Aumentar el tamaño del título
        },
        xaxis_title="Edad del Donante",
        yaxis_title="Cantidad de Donantes"
    )
    return fig


def calcular_min_max(df, EDAD):
    """Calcula los valores mínimo y máximo de una columna en un DataFrame."""
    min_val = df[EDAD].min()
    max_val = df[EDAD].max()
    return min_val, max_val

def mostrar_boxplot(df):


    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df['EDAD'],
        name="Distribución de Edades",
        boxpoints=False,  # No muestra los puntos
        marker_color="lightskyblue"  # Color azul claro para el boxplot
    ))

    # Configurar márgenes y tamaño para centrar
    fig.update_layout(
        width=500,  # Ajusta el ancho para centrar mejor
        height=500,  # Ajusta la altura
        margin=dict(l=50, r=50, t=60, b=60)  # Márgenes equilibrados
    )

    # Mostrar el gráfico centrado en Streamlit
    col1, col2, col3 = st.columns([1, 1, 1])  # Columnas, la del centro es la más grande
    with col2:
        st.plotly_chart(fig, use_container_width=False)

def extract_text_with_bold(pdf_path):
    """Extrae texto de un PDF e intenta identificar palabras en negrita."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if words:
                    for word in words:
                        font = word.get("fontname", "").lower()
                        text += f"**{word['text']}** " if "bold" in font else f"{word['text']} "
                text += "\n\n"  # Salto de línea entre páginas
        return text
    except FileNotFoundError:
        return None


def detectar_outliers(df):
    # Calcular cuartiles y rango intercuartílico (IQR)
    Q1 = df["EDAD"].quantile(0.25)
    Q3 = df["EDAD"].quantile(0.75)
    IQR = Q3 - Q1

    # Límites para detectar outliers
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Filtrar valores atípicos
    outliers = df[(df["EDAD"] < limite_inferior) | (df["EDAD"] > limite_superior)]

    # Mostrar el código en Streamlit
    codigo = '''Q1 = bancoDB["EDAD"].quantile(0.25)
Q3 = bancoDB["EDAD"].quantile(0.75)
IQR = Q3 - Q1

# Límites para detectar outliers
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = bancoDB[(bancoDB["EDAD"] < limite_inferior) | (bancoDB["EDAD"] > limite_superior)]
print("Número de valores atípicos:", len(outliers))'''

    st.markdown("### 📜 **Busqueda valores atipicos:**")
    st.code(codigo, language='python')

    # Mostrar el resultado en Streamlit
    st.markdown("### 📊 **Número de valores atípicos:**")
    st.write(f"🔍 **{len(outliers)} valores atípicos encontrados**")

    # Opcional: Mostrar los valores atípicos detectados
    if not outliers.empty:
        st.write("📋 **Valores atípicos detectados:**")
        st.dataframe(outliers)

def conteo_genero(df):
    # Calcular el conteo y la proporción de cada género
    conteo_genero = df['GENERO'].value_counts()

    # Crear gráfico de pastel con Plotly
    fig = px.pie(
        conteo_genero, 
        values=conteo_genero.values, 
        names=conteo_genero.index,
        title="Distribución por Género",
        color=conteo_genero.index,
        color_discrete_map={'M': 'blue', 'F': 'pink'}  # Ajusta los colores según las clases presentes
    )

    fig.update_traces(textinfo='percent+label')  # Mostrar porcentaje y etiqueta

    # Ajustar el diseño
    fig.update_layout(
        title={'text': "Distribución por Género", 'x': 0.45, 'xanchor': 'center', 'font': {'size': 24}}
    )

    # Mostrar en Streamlit
    st.plotly_chart(fig)

def promedio_edad(df):

    # Calcular el promedio de edad por género
    edad_promedio_genero = df.groupby('GENERO')['EDAD'].mean().reset_index()

    # Crear el gráfico de barras con barras más delgadas
    fig = px.bar(
        edad_promedio_genero, 
        x='GENERO', 
        y='EDAD', 
        color='GENERO', 
        text=edad_promedio_genero['EDAD'].round(1),  # Mostrar valores redondeados
        title='Edad Promedio por Género',
        labels={'EDAD': 'Edad Promedio', 'GENERO': 'Género'},
        color_discrete_sequence=['pink', 'blue']  # Colores personalizados
    )

    # Ajustar el diseño para hacer las barras más delgadas
    fig.update_traces(textposition='outside', textfont_size=12)
    fig.update_layout(xaxis_tickangle=0.5, showlegend=False, bargap=0.5)  # Aumentar bargap hace las barras más delgadas

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)


def genero_donante(df):
    # Crear tabla de contingencia
    contingencia_genero_donante = pd.crosstab(df['GENERO'], df['TIPO DONANTE'])
    
    # Convertir la tabla de contingencia a un DataFrame adecuado para Plotly
    df_plotly = contingencia_genero_donante.T.reset_index().melt(id_vars='TIPO DONANTE', var_name='Género', value_name='Cantidad')
    
    # Crear la gráfica de barras agrupadas con etiquetas
    fig = px.bar(df_plotly, 
                 x='TIPO DONANTE', 
                 y='Cantidad', 
                 color='Género', 
                 barmode='group', 
                 text='Cantidad',  # Agregar etiquetas con valores
                 color_discrete_map={'MASCULINO': 'blue', 'FEMENINO': 'pink'})  # Colores personalizados
    
    # Configurar el diseño
    fig.update_layout(
        title=dict(text='Comparación de Género por Tipo de Donante', x=0),
        xaxis_title='Tipo de Donante',
        yaxis_title='Número de Donantes',
        xaxis_tickangle=-45,  # Rotar etiquetas para mejor lectura
        legend_title='Género'
    )
    
    # Ajustar el tamaño de las etiquetas sobre las barras
    fig.update_traces(textposition='outside', textfont_size=10)
    
    # Mostrar la figura en Streamlit
    st.plotly_chart(fig)   
    st.table(contingencia_genero_donante)

def genero_gruposanguineo(df):
     # Crear tabla de contingencia
    contingencia_genero_sanguineo = pd.crosstab(df['GENERO'], df['GRUPO SANGUINEO']).reset_index()

    # Convertir la tabla a formato largo (melt) para Plotly
    df_melted = contingencia_genero_sanguineo.melt(id_vars='GENERO', var_name='GRUPO SANGUINEO', value_name='CANTIDAD')

    # Título de la aplicación
    #st.title("Análisis de Donantes: Género vs Grupo Sanguíneo")

    # Crear gráfico de barras
    fig = px.bar(
        df_melted, 
        x='GENERO', 
        y='CANTIDAD', 
        color='GRUPO SANGUINEO', 
        text='CANTIDAD',
        title='Distribución de Donantes por Género y Grupo Sanguíneo',
        labels={'CANTIDAD': 'Número de Donantes', 'GENERO': 'Género'},
        barmode='group',
        color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96', '#2CA02C']
    )

    # Ajustar diseño
    fig.update_traces(textposition='outside', textfont_size=10)
    fig.update_layout(xaxis_tickangle=0, bargap=0.3)

    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)


def grafico_diferido(df):
    #st.title('Frecuencia de Tipos de Diferido')
    
    # Contar la frecuencia de cada tipo de diferido
    diferido_counts = df['TIPO DE DIFERIDO'].value_counts().reset_index()
    diferido_counts.columns = ['TIPO DE DIFERIDO', 'CANTIDAD']
    
    # Crear gráfico de barras
    fig = px.bar(diferido_counts, 
                 x='TIPO DE DIFERIDO', 
                 y='CANTIDAD', 
                 text='CANTIDAD',
                 title='Frecuencia de Tipos de Diferido',
                 labels={'TIPO DE DIFERIDO': 'Tipo de Diferido', 'CANTIDAD': 'Número de Donantes'},
                 color='TIPO DE DIFERIDO',  # Color por categoría
                 color_discrete_sequence=px.colors.qualitative.Set2)
    
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, bargap=0.3)
    
    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)

def grafico_donante(df):
    #st.title('Frecuencia de Tipos de Donante')
    
    # Contar la frecuencia de cada tipo de donante
    donante_counts = df['TIPO DONANTE'].value_counts().reset_index()
    donante_counts.columns = ['TIPO DONANTE', 'CANTIDAD']
    
    # Crear gráfico de barras
    fig = px.bar(donante_counts, 
                 x='TIPO DONANTE', 
                 y='CANTIDAD', 
                 text='CANTIDAD',
                 title='Frecuencia de Tipos de Donante',
                 labels={'TIPO DONANTE': 'Tipo de Donante', 'CANTIDAD': 'Número de Donantes'},
                 color='TIPO DONANTE',  
                 color_discrete_sequence=px.colors.qualitative.Pastel1)
    
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, bargap=0.3)
    
    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)

def grafico_donacion(df):
    #st.title('Frecuencia de Tipos de Donación')
    
    # Contar la frecuencia de cada tipo de donación
    donacion_counts = df['TIPO DE DONACION'].value_counts().reset_index()
    donacion_counts.columns = ['TIPO DE DONACION', 'CANTIDAD']
    
    # Crear gráfico de barras
    fig = px.bar(donacion_counts, 
                 x='TIPO DE DONACION', 
                 y='CANTIDAD', 
                 text='CANTIDAD',
                 title='Frecuencia de Tipos de Donación',
                 labels={'TIPO DE DONACION': 'Tipo de Donación', 'CANTIDAD': 'Número de Donantes'},
                 color='TIPO DE DONACION',  
                 color_discrete_sequence=px.colors.qualitative.Dark2)
    
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, bargap=0.3)
    
    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)


def donanteVSdonacion(df):
    # Crear tabla de contingencia
    x = pd.crosstab(df['TIPO DONANTE'], df['GRUPO SANGUINEO']).reset_index()
    df_melted = x.melt(id_vars='TIPO DONANTE', var_name='GRUPO SANGUINEO', value_name= 'CANTIDAD')


    st.subheader("Tabla de contingencia en formato largo:")
    st.dataframe(df_melted)

    return df_melted  # Retorna el DataFrame si necesitas usarlo en otro lugar


def grupos_sanguineos(df):
    # Contar la cantidad de cada grupo sanguíneo
    grupo_sanguineo_counts = df['GRUPO SANGUINEO'].value_counts().reset_index()
    grupo_sanguineo_counts.columns = ['Grupo Sanguíneo', 'Cantidad']
    
    # Crear gráfico de barras
    fig = px.bar(grupo_sanguineo_counts, 
                 x='Grupo Sanguíneo', 
                 y='Cantidad', 
                 text='Cantidad',
                 color='Grupo Sanguíneo',
                 title="Distribución de Grupos Sanguíneos")
    
    # Ajustar etiquetas
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title="Grupo Sanguíneo", yaxis_title="Número de Donantes")
    
    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)


def grafico_rh(df):
    # Crear tabla de contingencia
    contingencia_genero_sanguineo = pd.crosstab(df['GENERO'], df['GRUPO SANGUINEO']).reset_index()
    
    # Crear gráfico de barras con una barra separada por grupo sanguíneo
    fig = px.bar(contingencia_genero_sanguineo.melt(id_vars='GENERO', var_name='Grupo Sanguíneo', value_name='Cantidad'),
                 x='Grupo Sanguíneo', 
                 y='Cantidad', 
                 color='GENERO',
                 title="Grupo Sanguíneo por Género",
                 labels={'Cantidad': 'Número de Donantes', 'GENERO': 'Género'},
                 barmode='group')
    
    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)

def gruposanguineoVSdonacion(df):
    # Crear tabla de contingencia
    contingencia_sangre_donacion = pd.crosstab(df['GRUPO SANGUINEO'], df['TIPO DE DONACION']).reset_index()
    
    # Crear gráfico de barras apiladas
    fig = px.bar(contingencia_sangre_donacion.melt(id_vars='GRUPO SANGUINEO', var_name='Tipo de Donación', value_name='Cantidad'),
                 x='GRUPO SANGUINEO', 
                 y='Cantidad', 
                 color='Tipo de Donación',
                 title="Grupo Sanguíneo vs. Tipo de Donación",
                 labels={'Cantidad': 'Número de Donantes', 'GRUPO SANGUINEO': 'Grupo Sanguíneo'},
                 barmode='stack')
    
    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)

def edad_por_tipo_donacion(df):
    st.title("Distribución de edad según tipo de donación")
    
    # Crear gráfico de caja
    fig = px.box(df, 
                 x="TIPO DE DONACION", 
                 y="EDAD", 
                 color="TIPO DE DONACION",
                 #title="Distribución de Edad según Tipo de Donación",
                 width=900, height=600)
    
    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)

def mostrar_grafico_grupo_rh(df):
    # Crear una nueva columna que combine Grupo Sanguíneo y Rh
    df['GRUPO_RH'] = df['GRUPO SANGUINEO'] + " " + df['ANTIGENO RH-D']
    
    # Contar la cantidad de cada combinación de Grupo Sanguíneo + Rh
    grupo_rh_counts = df['GRUPO_RH'].value_counts().reset_index()
    grupo_rh_counts.columns = ['Grupo Sanguíneo + Rh', 'Cantidad']
    
    # Crear gráfico de barras
    fig = px.bar(grupo_rh_counts, 
                 x='Grupo Sanguíneo + Rh', 
                 y='Cantidad', 
                 text='Cantidad',
                 color='Grupo Sanguíneo + Rh',
                 title="Distribución de Grupos Sanguíneos + Rh")
    
    # Ajustar etiquetas
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title="Grupo Sanguíneo + Rh", yaxis_title="Número de Donantes")
    
    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)

def mostrar_grafico_grupo_rh_genero(df):
    # Crear tabla de contingencia
    contingencia_grupo_rh_genero = pd.crosstab(df['GENERO'], df['GRUPO_RH'])
    
    # Convertir la tabla en un formato adecuado para Plotly
    contingencia_grupo_rh_genero = contingencia_grupo_rh_genero.reset_index()
    df_melted = contingencia_grupo_rh_genero.melt(id_vars=['GENERO'], var_name='GRUPO_RH', value_name='Cantidad')
    
    # Crear gráfico de barras separadas por Grupo Sanguíneo + Rh
    fig = px.bar(df_melted, 
                 x='GENERO', 
                 y='Cantidad', 
                 color='GRUPO_RH',
                 title="Grupo Sanguíneo + Rh por Género",
                 labels={'Cantidad': 'Número de Donantes', 'GENERO': 'Género'},
                 barmode='group')
    
    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)


def lugar_donacion(df):
    #st.title("Frecuencia de Donaciones por Lugar")
    
    # Contar la cantidad de donaciones por lugar
    lugares_counts = df['LUGAR DE DONACION'].value_counts().reset_index()
    lugares_counts.columns = ['Lugar de Donación', 'Cantidad']
    
    # Crear gráfico de barras
    fig = px.bar(lugares_counts, 
                 x='Lugar de Donación', 
                 y='Cantidad', 
                 text='Cantidad',
                 color='Lugar de Donación',
                 title="Frecuencia de Donaciones por Lugar")
    
    # Ajustar etiquetas
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title="Lugar de Donación", 
                      yaxis_title="Número de Donaciones", 
                      xaxis={'categoryorder':'total descending'})
    
    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)

def donante_lugar_donacion(df):
    #st.title("Distribución de Tipo de Donante por Lugar de Donación")
    # Crear tabla de contingencia
    contingencia_lugar_donante = pd.crosstab(df['LUGAR DE DONACION'], df['TIPO DONANTE'])
    
    # Crear gráfico de barras agrupadas
    fig = px.bar(contingencia_lugar_donante, 
                 x=contingencia_lugar_donante.index, 
                 y=contingencia_lugar_donante.columns, 
                 title="Distribución de Tipo de Donante por Lugar de Donación",
                 labels={'value': 'Número de Donantes', 'LUGAR DONACION': 'Lugar de Donación'},
                 barmode='group')
    
    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)


def grafico_grupo_sanguineo_por_lugar(df):
    #st.title("Distribución de Grupo Sanguíneo por Lugar de Donación")
    # Crear tabla de contingencia
    contingencia_lugar_sangre = pd.crosstab(df['LUGAR DE DONACION'], df['GRUPO SANGUINEO'])
    
    # Crear gráfico de barras agrupadas
    fig = px.bar(contingencia_lugar_sangre, 
                 x=contingencia_lugar_sangre.index, 
                 y=contingencia_lugar_sangre.columns, 
                 title="Distribución de Grupo Sanguíneo por Lugar de Donación",
                 labels={'value': 'Número de Donantes', 'LUGAR DONACION': 'Lugar de Donación'},
                 barmode='group')
    
    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)

def edad_por_tipo_donante(df):
    #st.title("Distribución de Edad según Tipo de Donante")
    
    # Crear gráfico de caja
    fig = px.box(df, 
                 x="TIPO DONANTE", 
                 y="EDAD", 
                 color="TIPO DONANTE",
                 title="Distribución de Edad según Tipo de Donante"
                 #width=400,height=500)
            )
    # Mostrar gráfico en Streamlit
    st.plotly_chart(fig)

def predecir_donaciones(df):
    # Convertir fechas
    df["FECHA DONACION"] = pd.to_datetime(df["FECHA DONACION"])
    
    # Contar donaciones por mes
    donaciones_por_mes = df.resample("ME", on="FECHA DONACION").size().reset_index()
    donaciones_por_mes.columns = ["ds", "y"]  # Prophet usa columnas 'ds' (fecha) y 'y' (valor)
    
    # Crear y entrenar el modelo
    modelo = Prophet()
    modelo.fit(donaciones_por_mes)
    
    # Definir la fecha inicial para predicciones (enero 2025)
    fecha_inicio_prediccion = pd.to_datetime("2025-01-01")
    
    # Calcular cuántos meses predecir desde la última fecha del dataset hasta 2027 (3 años de predicción)
    ultima_fecha = donaciones_por_mes["ds"].max()
    meses_a_predecir = ((2027 - 2025) * 12) + 12  # Hasta diciembre de 2027
    
    # Generar fechas futuras a partir de enero 2025
    futuro = modelo.make_future_dataframe(periods=meses_a_predecir, freq="ME")
    futuro = futuro[futuro["ds"] >= fecha_inicio_prediccion]  # Filtrar solo desde enero 2025
    
    # Hacer predicciones
    predicciones = modelo.predict(futuro)
    
    # Gráfica con Plotly en Streamlit
    fig = px.line(predicciones, x="ds", y="yhat", title="Predicción de Donaciones desde Enero 2025")
    
    # Mostrar la gráfica en Streamlit
    st.plotly_chart(fig)
    
    return predicciones




