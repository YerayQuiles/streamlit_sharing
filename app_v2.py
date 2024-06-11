import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import speech_recognition as sr
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Descargar los datos de VADER para el análisis de sentimientos
nltk.download('vader_lexicon')

st.set_page_config(layout="wide")


# Definir las preguntas
questions = [
    {"question": "Durante la última semana, ¿has tenido poco interés o placer en hacer cosas?", "options": ["---", "Siempre", "Muy a menudo", "Algunas veces", "Rara vez", "Nunca"], "text_input": True},
    {"question": "Durante la última semana, ¿te has sentido tan deprimido, aburrido o sin ánimo que incluso pensaste en terminar tu vida?", "options": ["---", "Siempre", "Muy a menudo", "Algunas veces", "Rara vez", "Nunca"], "text_input": True},
    {"question": "Durante la última semana, ¿has tenido dificultades para concentrarte en cosas o para tomar decisiones?", "options": ["---", "Siempre", "Muy a menudo", "Algunas veces", "Rara vez", "Nunca"], "text_input": True},
    {"question": "Durante la última semana, ¿has tenido dificultades para dormir, como tener problemas para quedarte dormido, despertarte temprano y no poder volver a dormir, o soñar mucho?", "options": ["---", "Siempre", "Muy a menudo", "Algunas veces", "Rara vez", "Nunca"], "text_input": True},
    {"question": "¿Hay algo más que quieras compartir con nosotros?", "options": [], "text_input": True}
]

# Cargar el logo desde un archivo

logo_path = "C:/Users/yeray/OneDrive/Escritorio/NexiHealth/PHQ-9/logo.png"  # Reemplaza con la ruta correcta de tu logo
# Centrar la imagen del logo dentro de una columna
col1, col2, col3 = st.columns([1, 3, 1])  # Columnas vacías a los lados para centrar

with col2:
    st.image(logo_path, width=400 )


# ______________________________________________________________________________________________________________________________________________
# ____________________________________________________FORMULARIO________________________________________________________________________________
# ______________________________________________________________________________________________________________________________________________

def show_personal_info_form():
    st.title("Información Personal")
    st.info("Por favor, proporciona tu información personal antes de responder las preguntas.")
    
    # Crear un formulario para los datos personales
    with st.form(key='personal_info_form'):
        name = st.text_input("Nombre")
        last_name = st.text_input("Apellidos")
        age = st.number_input("Edad", min_value=0, max_value=120)
        submit_button = st.form_submit_button(label='Continuar')
        
        if submit_button:
            if name and last_name and age:
                st.session_state.personal_info = {
                    "name": name,
                    "last_name": last_name,
                    "age": age
                }
                st.session_state.show_questions = True
                st.experimental_rerun()
            else:
                st.warning("Por favor, completa todos los campos antes de continuar.")

def show_form():
    st.title("Formulario")
    st.info(
        f"Hola {st.session_state.personal_info['name']}, responde las siguientes preguntas con sinceridad. Utiliza el recuadro de información adicional para escribir cualquier información que creas que puede ser relevante para que intentemos ayudarte de la mejor forma posible."
    )
    st.markdown("---")  # Línea separadora

    # Iniciar una lista para almacenar las respuestas
    responses = []

    # Iterar sobre cada pregunta
    for index, q in enumerate(questions):
        question_text = q["question"]
        options = q["options"]
        text_input = q["text_input"]

        st.subheader(question_text)
        if text_input:
            selected_option = st.selectbox(f"Selecciona:", options, key=f"select_{index}")
            col1, col2 = st.columns([3, 1])
            text_response = st.text_area(f"¿Puedes contarnos algo más respecto a la pregunta?:", key=f"text_input_{index}")
            if f"text_input_{index}" in st.session_state:
                text_response = st.session_state[f"text_input_{index}"]
            responses.append({"question": question_text, "selected_option": selected_option, "text_response": text_response})
        else:
            selected_option = st.selectbox("Selecciona:", options, key=f"select_{index}")
            if selected_option == "---":
                st.warning("Por favor, seleccione una opción válida.")
            else:
                responses.append({"question": question_text, "selected_option": selected_option, "text_response": ""})

    # Validar si todas las opciones fueron seleccionadas
    if all(response["selected_option"] != "---" for response in responses):
        # Agregar un botón de enviar
        if st.button("Enviar"):
            # Guardar las respuestas en el estado de la sesión
            st.session_state.responses = responses
            # Cambiar la vista a la página de informe
            st.session_state.show_report = True
            st.experimental_rerun()
    else:
        st.warning("Por favor, complete todas las preguntas antes de continuar.")

# ______________________________________________________________________________________________________________________________________________
# _______________________________________________________INFORME________________________________________________________________________________
# ______________________________________________________________________________________________________________________________________________
def show_report():
    st.title("Informe de Respuestas")
    st.write("Gracias por tus respuestas. Aquí está el resumen de tus respuestas:")
    
    # Mostrar la tabla con las respuestas
    df = pd.DataFrame(st.session_state.responses)
    st.write(df, wide_mode=True)
    st.info("Puedes ampliar, filtrar o descargar la tabla en formato CSV pinchando sobre los iconos de la esquina superior derecha")

    
    
    # Calcular la puntuación PHQ-9
    phq_score = sum([questions[i]["options"].index(st.session_state.responses[i]["selected_option"]) for i in range(len(questions) - 1)])

    # Calcular el grado de depresión basado en la puntuación PHQ-9
    if phq_score <= 4:
        depression_level = "Ninguna o mínima"
        background_color = "green"
        text_color = "white"
    elif phq_score <= 9:
        depression_level = "Leve"
        background_color = "lightgreen"
        text_color = "black"
    elif phq_score <= 14:
        depression_level = "Moderada"
        background_color = "yellow"
        text_color = "black"
    elif phq_score <= 19:
        depression_level = "Moderadamente severa"
        background_color = "orange"
        text_color = "white"
    else:
        depression_level = "Severa"
        background_color = "red"
        text_color = "white"

    # Mostrar el grado de depresión
    st.subheader(f"El grado de depresión basado en la puntuación PHQ-9 es: {depression_level}")

    # Crear el mensaje con el color correspondiente al grado de depresión
    # Crear el mensaje con el color correspondiente al grado de depresión
    message = f"""
    <div style="width: 200px; height: 50px; margin: 0 auto; background-color: {background_color}; color: {text_color}; padding: 5px; border-radius: 5px; display: flex; align-items: center; justify-content: center;">
        <p style="font-size: 24px; font-weight: bold; margin: 0; text-align: center;">{depression_level}</p>
    </div>
    """

    # Mostrar el mensaje en el recuadro con el color correspondiente
    st.markdown(message, unsafe_allow_html=True)



    # ______________________________________________________________________________________________________________________________________________
    # _________________________________________________GRAFICO PHQ-9________________________________________________________________________________
    # ______________________________________________________________________________________________________________________________________________
    # Graficar el indicador de PHQ-9
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=phq_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Puntuación PHQ-9", 'font': {'size': 24}},
        number={'font': {'size': 36}},
        gauge={
            'axis': {'range': [None, 27], 'tickwidth': 2, 'tickcolor': "black"},
            'bar': {'color': "white"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 5], 'color': "green"},
                {'range': [5, 10], 'color': "lightgreen"},
                {'range': [10, 15], 'color': "yellow"},
                {'range': [15, 20], 'color': "orange"},
                {'range': [20, 27], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': phq_score
            }
        }
    ))
    st.plotly_chart(fig)
    
    # ______________________________________________________________________________________________________________________________________________
    # ______________________________________________NUBE DE PALABRAS________________________________________________________________________________
    # ______________________________________________________________________________________________________________________________________________
    # Generar la nube de palabras a partir de las respuestas adicionales
    additional_texts = " ".join([response["text_response"] for response in st.session_state.responses if response["text_response"]])
    
    if additional_texts:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(additional_texts)
        
        st.subheader("Nube de palabras de la información adicional:")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        with st.container():
            st.pyplot(fig)
            
        # Análisis de sentimiento
        sid = SentimentIntensityAnalyzer()
        sentiments = [sid.polarity_scores(text) for text in additional_texts.split()]

        # Calcular el promedio de los sentimientos
        avg_sentiment = {
            "neg": sum(s["neg"] for s in sentiments) / len(sentiments),
            "neu": sum(s["neu"] for s in sentiments) / len(sentiments),
            "pos": sum(s["pos"] for s in sentiments) / len(sentiments),
            "compound": sum(s["compound"] for s in sentiments) / len(sentiments)
        }

        st.subheader("Análisis de sentimiento de la información adicional:")
        
        # Mostrar los puntajes promedio de sentimiento
        st.write(f"Negativo: {avg_sentiment['neg']:.2f}")
        st.write(f"Neutro: {avg_sentiment['neu']:.2f}")
        st.write(f"Positivo: {avg_sentiment['pos']:.2f}")
        st.write(f"Compuesto: {avg_sentiment['compound']:.2f}")

        # Crear una gráfica de barras con Plotly
        fig = go.Figure(data=[
            go.Bar(name='Negativo', x=['Sentimiento'], y=[avg_sentiment['neg']], marker_color='red'),
            go.Bar(name='Neutro', x=['Sentimiento'], y=[avg_sentiment['neu']], marker_color='gray'),
            go.Bar(name='Positivo', x=['Sentimiento'], y=[avg_sentiment['pos']], marker_color='green'),
            go.Bar(name='Compuesto', x=['Sentimiento'], y=[avg_sentiment['compound']], marker_color='blue')
        ])

        # Actualizar el diseño de la gráfica para una mejor visualización
        fig.update_layout(barmode='group', title="Resultados del Análisis de Sentimiento", xaxis_title="Tipo de Sentimiento", yaxis_title="Promedio")
        with st.container():
            st.plotly_chart(fig, use_container_width=True)

# Inicializar el estado de la sesión si no está definido
if 'show_report' not in st.session_state:
    st.session_state.show_report = False

if 'show_questions' not in st.session_state:
    st.session_state.show_questions = False

if 'personal_info' not in st.session_state:
    st.session_state.personal_info = False

# Mostrar el formulario de información personal si no se ha completado
if not st.session_state.personal_info:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        show_personal_info_form()
# Mostrar el formulario de preguntas si se ha completado el formulario de información personal
elif not st.session_state.show_report:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        show_form()
# Mostrar el informe si se han completado ambos formularios
else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:

     show_report()
