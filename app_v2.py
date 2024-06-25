import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Descargar recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Descargar los datos de VADER para el análisis de sentimientos
nltk.download('vader_lexicon')
 
st.set_page_config(layout="wide")

# Definir las preguntas
questions_anxiety = [
    {
        "question": "¿Has tenido un estado de ánimo ansioso? Como preocupación, irritabilidad o anticipación de lo peor.",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido sensación de tensión, imposibilidad de relajarse, llanto fácil o temblores?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido temores, por ejemplo a la oscuridad, a quedarse solo o a las multitudes?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido insomnio?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido dificultad para concentrarte o mala memoria?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido un estado de ánimo depresivo? Como pérdida de interes, insatisfacción o cambios de humor",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido dolores, molestias musculares o rigidez?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido zumbidos de oídos, visión borrosa, sofocos o escalofrios?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido palpitaciones, dolor en el pecho o sensación de hormigueo?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido dificultad para respirar, suspiros o sensación de opresión en el pecho?.",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido náuseas, vómitos, diarrea, dolor abdominal, sensación de hinchazón?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido un aumento de la frecuencia urinaria, dolor al orinar, falta de deseo sexual, disfunción eréctil?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido sequedad de boca, sudoración, mareos, sofocos, sensación de frío?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Consideras que en una conversación te encuentras tenso, inquieto, con tics, taquicardia, sacudidas o sudoración excesivas?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Hay algo más que quieras compartir con nosotros sobre estas preguntas?. Nos ayudará a proporcionarte la mejor atención.?",
        "options": [],
        "text_input": True
    }
]
questions_depression = [
    {
        "question": "¿Has tenido un estado de ánimo depresivo? Como tristeza, desesperanza, sentimiento de inutilidad.",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido sentimientos de culpa?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido ideas de suicidio?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido dificultad para quedarte dormido?",
        "options": ["---", "Ninguno", "Leve", "Grave "],
        "text_input": True
    },
    {
        "question": "¿Has tenido insomnio durante la noche? Como estar desvelado o haberte despertado muchas veces. ",
        "options": ["---", "Ninguno", "Leve", "Grave "],
        "text_input": True
    },
    {
        "question": "¿Has tenido dificultad para volver a dormirte de madrugada?",
        "options": ["---", "Ninguno", "Leve", "Grave "],
        "text_input": True
    },
    {
        "question": "¿Has tenido problemas en el trabajo o realizando otras actividades' Como sentimientos de incapacidad, de fatiga, debilidad, o incluso dejar de realizar estas actividades.",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido lentitud de pensamiento, lentitud al hablar o concentración disminuida?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has sentido agitación?",
        "options": ["---", "Ninguno", "Leve", "Grave "],
        "text_input": True
    },
    {
        "question": "¿Has tenido ansiedad? Irritabilidad, preocupación, o tendencia a expresar tus temores?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido problemas gastroinstentinales, sequedad de boca, palpiraciones, hiperventilación, frecuencia en la micción o sudoración?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido pérdida de apetito, sensación de pesadez en el abdomen o dificultadd para comer?",
        "options": ["---", "Ninguno", "Leve", "Grave "],
        "text_input": True
    },
    {
        "question": "¿Has tenido pesadez en las extremidades, espalda o cabeza?",
        "options": ["---", "Ninguno", "Leve", "Grave "],
        "text_input": True
    },
    {
        "question": "¿Has tenido disminución de la líbido o transtornos menstruales?",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido hipocondría? .",
        "options": ["---", "Ninguno", "Leve", "Moderado", "Grave", "Muy grave"],
        "text_input": True
    },
    {
        "question": "¿Has tenido pérdidas de peso?",
        "options": ["---", "Ninguno", "Leve", "Grave "],
        "text_input": True
    },
    {
        "question": "¿Consideras que tienes depresión?",
        "options": ["---", "Ninguno", "Leve", "Grave "],
        "text_input": True
    },
    {
        "question": "¿Hay algo más que quieras compartir con nosotros sobre tus sentimientos de ansiedad?",
        "options": [],
        "text_input": True
    }
]
# Combining the two lists
combined_questions = questions_anxiety + questions_depression

# Cargar el logo desde un archivo
logo_path = "./logo.png"
st.logo(logo_path, link="https://nexihealth.es/")
# Columnas con espacio a los lados para centrar
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])  

# Mostrar el logo en la columna central
with col3:
    st.image(logo_path, use_column_width=True)

# ______________________________________________________________________________________________________________________________________________
# ____________________________________________________FORMULARIO________________________________________________________________________________
# ______________________________________________________________________________________________________________________________________________

def show_password_prompt():
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.info("Si estás interesado en probar la herramienta y saber más sobre ella no dudes en pedirnos acceso escribiendo a info@nexihealth.es")

        st.title("Ingresa la contraseñaA")
        password = st.text_input("Contraseña", type="password")
        if st.button("Enviar"):
            if password == "Nexi":
                st.session_state.password_correct = True
                st.experimental_rerun()
            else:
                st.error("Contraseña incorrecta, por favor intente nuevamente")
                
def show_personal_info_form():
    st.title("Información personal")
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
    st.title("Formularios")
    st.info(
        f"Hola {st.session_state.personal_info['name']}, responde las siguientes preguntas con sinceridad. Utiliza el recuadro de 'Añadir información' para escribir cualquier información extra que creas que puede ser relevante para que intentemos ayudarte de la mejor forma posible."
    )
    st.markdown("---")  # Línea separadora

    # Iniciar una lista para almacenar las respuestas
    responses = []
    # Iterar sobre cada pregunta
    for index, q in enumerate(combined_questions):
        question_text = q["question"]
        options = q["options"]
        text_input = q["text_input"]

        st.markdown(f"##### {question_text}")
        
        # Crear una fila para el select box y el checkbox
        select_col, checkbox_col = st.columns([3, 1])

        # Select box para la opción
        selected_option = select_col.selectbox("Selecciona:", options, key=f"select_{index}")
        
        # Checkbox para mostrar el campo de texto
        show_text_area = checkbox_col.toggle("Añadir información", key=f"show_text_{index}")
        
        # Campo de texto condicional
        text_response = None
        if show_text_area:
            text_response = st.text_area(f"¿Puedes contarnos algo más respecto a la pregunta?:", key=f"text_input_{index}")
        
        st.write("")

        # Guardar las respuestas
        responses.append({"question": question_text, "selected_option": selected_option, "text_response": text_response})
        
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
        
    # Crear tres columnas para alinear el botón "Atrás" a la derecha
    _, _, col_atras = st.columns([1, 10, 1])
    
    with col_atras:
        if st.button("Atrás"):
            st.session_state.personal_info = False
            st.experimental_rerun()
        

# ______________________________________________________________________________________________________________________________________________
# _______________________________________________________INFORME________________________________________________________________________________
# ______________________________________________________________________________________________________________________________________________
def show_report():
    st.title("Informe de Respuestas")
    st.info("Añadir aqui un texto informativo sobre el informe.")

    # Verificar si la respuesta a "¿Has tenido ideas de suicidio?" es "Grave" o "Muy grave"
    for response in st.session_state.responses:
        if response["question"] == "¿Has tenido ideas de suicidio?" and response["selected_option"] == "Grave":
            st.error("ALERTA: Riesgo de suicidio.")
            break
        if response["question"] == "¿Has tenido ideas de suicidio?" and response["selected_option"] == "Muy grave":
            st.error("ALERTA: Riesgo alto de suicidio.")
            break

    # Filtrar las respuestas relacionadas con la ansiedad
    responses_anxiety = [response for response in st.session_state.responses if response["question"] in [q["question"] for q in questions_anxiety]]
    # Filtrar las respuestas relacionadas con la depresión
    responses_depression = [response for response in st.session_state.responses if response["question"] in [q["question"] for q in questions_depression]]

    # Mostrar la tabla con las respuestas de la ansiedad si se selecciona el checkbox
    df_anxiety = pd.DataFrame(responses_anxiety)
    if st.toggle("Mostrar Respuestas sobre Ansiedad"):
        st.subheader("Respuestas sobre Ansiedad")
        st.write(df_anxiety, use_container_width=True)
        st.info("Puedes ampliar, filtrar o descargar las tablas en formato CSV pinchando sobre los iconos de la esquina superior derecha")

    # Mostrar la tabla con las respuestas de la depresión si se selecciona el checkbox
    df_depression = pd.DataFrame(responses_depression)
    if st.toggle("Mostrar Respuestas sobre Depresión"):
        st.subheader("Respuestas sobre Depresión")
        st.write(df_depression, use_container_width=True)
        st.info("Puedes ampliar, filtrar o descargar las tablas en formato CSV pinchando sobre los iconos de la esquina superior derecha")

    # Crear un diccionario que mapea los valores de "selected_option" a números
    option_map = {"Ninguno": 0, "Leve": 1, "Moderado": 2, "Grave": 3, "Muy grave": 4, "Grave ": 2}

    # Aplicar el mapeo a la columna "selected_option" de la tabla df_depression
    df_depression["selected_option"] = df_depression["selected_option"].map(option_map)
    
    # _______________________________________________________ANSIEDAD________________________________________________________________________________
    # Calcular la puntuación HAM-A
    ham_a_score = sum([0 if st.session_state.responses[i]["selected_option"] == "Ninguno" else questions_anxiety[i]["options"].index(st.session_state.responses[i]["selected_option"]) - 1 for i in range(len(questions_anxiety) - 1)])

    # Calcular el grado de ansiedad basado en la puntuación HAM-A
    if ham_a_score <= 17:
        anxiety_level = "Normal"
        background_color = "green"
        text_color = "white"
    elif ham_a_score <= 24:
        anxiety_level = "Leve"
        background_color = "lightgreen"
        text_color = "black"
    elif ham_a_score <= 30:
        anxiety_level = "Moderada"
        background_color = "yellow"
        text_color = "black"
    elif ham_a_score <= 38:
        anxiety_level = "Severa"
        background_color = "orange"
        text_color = "white"
    else:
        anxiety_level = "Muy severa"
        background_color = "red"
        text_color = "white"

    col1, col2  = st.columns([1, 1])
    with col1:

        # Mostrar el grado de ansiedad
        st.subheader(f"El grado de ansiedad basado en la puntuación HAM-A es: {anxiety_level}")

        # Crear el mensaje con el color correspondiente al grado de ansiedad
        message = f"""
        <div style="width: 200px; height: 50px; margin: 0 auto; background-color: {background_color}; color: {text_color}; padding: 5px; border-radius: 5px; display: flex; align-items: center; justify-content: center;">
            <p style="font-size: 24px; font-weight: bold; margin: 0; text-align: center;">{anxiety_level}</p>
        </div>
        """

        # Mostrar el mensaje en el recuadro con el color correspondiente
        st.markdown(message, unsafe_allow_html=True)
        
        # ______________________________________________________________________________________________________________________________________________
        # _________________________________________________GRAFICO ANSIEDAD________________________________________________________________________________
        # ______________________________________________________________________________________________________________________________________________
        # Graficar el indicador de PHQ-9
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ham_a_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Puntuación HAM-A", 'font': {'size': 24}},
            number={'font': {'size': 36}},
            gauge={
                'axis': {'range': [None, 50], 'tickwidth': 2, 'tickcolor': "black"},
                'bar': {'color': "white"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 17], 'color': "green"},
                    {'range': [17, 24], 'color': "lightgreen"},
                    {'range': [24, 30], 'color': "yellow"},
                    {'range': [30, 38], 'color': "orange"},
                    {'range': [38, 50], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': ham_a_score
                }
            }
        ))
        st.plotly_chart(fig)
        
        
        
        # Calcular indice de ansiedad psquica o somática
        questions_ansiedad_psiquica = [
        "¿Has tenido un estado de ánimo ansioso? Como preocupación, irritabilidad o anticipación de lo peor.",
        "¿Has tenido sensación de tensión, imposibilidad de relajarse, llanto fácil o temblores?",
        "¿Has tenido temores, por ejemplo a la oscuridad, a quedarse solo o a las multitudes?",
        "¿Has tenido insomnio?",
        "¿Has tenido dificultad para concentrarte o mala memoria?",
        "¿Has tenido un estado de ánimo depresivo? Como pérdida de interes, insatisfacción o cambios de humor",
        "¿Consideras que en una conversación te encuentras tenso, inquieto, con tics, taquicardia, sacudidas o sudoración excesivas?"
        ]
        ansiedad_psiquica_index = 0

        for response in st.session_state.responses:
            if response["question"] in questions_ansiedad_psiquica:
                ansiedad_psiquica_index += option_map.get(response["selected_option"], 0)

        print(f"Índice de ansiedad psíquica: {ansiedad_psiquica_index}")


        questions_ansiedad_somatica = [
            "¿Has tenido dolores, molestias musculares o rigidez?",
            "¿Has tenido zumbidos de oídos, visión borrosa, sofocos o escalofrios?",
            "¿Has tenido palpitaciones, dolor en el pecho o sensación de hormigueo?",
            "¿Has tenido dificultad para respirar, suspiros o sensación de opresión en el pecho?.",
            "¿Has tenido náuseas, vómitos, diarrea, dolor abdominal, sensación de hinchazón?",
            "¿Has tenido un aumento de la frecuencia urinaria, dolor al orinar, falta de deseo sexual, disfunción eréctil?",
            "¿Has tenido sequedad de boca, sudoración, mareos, sofocos, sensación de frío?"
            
        ]
        ansiedad_somatica_index = 0

        for response in st.session_state.responses:
            if response["question"] in questions_ansiedad_somatica:
                ansiedad_somatica_index += option_map.get(response["selected_option"], 0)

        print(f"Índice de ansiedad psíquica: {ansiedad_somatica_index}")
        
        #st.info(f"El grado de ansiedad psíquica es: {ansiedad_psiquica_index}")
        #st.info(f"El grado de ansiedad somática es: {ansiedad_somatica_index}")
        # Cálculo de porcentajes
        max_index = 28  # Valor máximo para ambos índices
        porcentaje_ansiedad_psiquica = round((ansiedad_psiquica_index / max_index) * 100, 1)
        porcentaje_ansiedad_somatica = round((ansiedad_somatica_index / max_index) * 100, 1)
       # Mostrar información con Streamlit
        st.info(f"El grado de ansiedad psíquica es: {porcentaje_ansiedad_psiquica}%")
        st.info(f"El grado de ansiedad somática es: {porcentaje_ansiedad_somatica}%")

        # Colores personalizados
        colors1 = ['gray'] * 3  # Inicialmente gris para todos los sectores
        colors1[0] = 'blue'    # Primer sector (psíquica) en verde
        colors1[1] = 'gray'     # Segundo sector (somática) en azul
        
        # Colores personalizados
        colors2 = ['gray'] * 3  # Inicialmente gris para todos los sectores
        colors2[0] = 'green'    # Primer sector (psíquica) en verde
        colors2[1] = 'gray'     # Segundo sector (somática) en azul

        # Crear subplots: utilizar 'domain' para el tipo de subplot de Pie
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

        # Añadir trazos de Pie para cada tipo de ansiedad
        fig.add_trace(go.Pie(
            values=[porcentaje_ansiedad_psiquica, 100 - porcentaje_ansiedad_psiquica], 
            labels=["Psíquica", "No definida"],
            marker=dict(colors=colors1),
            name="",
            legendgroup="group1"),  # Definir grupo de leyendas para agrupar este y otros
            1, 1)

        fig.add_trace(go.Pie(
            values=[porcentaje_ansiedad_somatica, 100 - porcentaje_ansiedad_somatica],
            labels=["Somática", "No definida"],
            marker=dict(colors=colors2),
            name="",
            legendgroup="group1"),  # Mismo grupo de leyendas para agrupar este y otros
            1, 2)

        # Usar `hole` para crear un gráfico de tipo donut
        fig.update_traces(hole=.4, hoverinfo="label+percent", textinfo='label+percent')

        # Actualizar el diseño del gráfico
        fig.update_layout(
            title_text="Tipo de Ansiedad",
        )

        # Mostrar el gráfico utilizando Streamlit
        st.plotly_chart(fig)
       
        
    
    # _______________________________________________________DEPRESIÓN________________________________________________________________________________
    # Calcular la puntuación HAM-D

    # Sumar todos los valores de la columna "selected_option" en la variable ham_d_score
    ham_d_score = df_depression["selected_option"].sum()    
    print(ham_a_score)
    print(ham_d_score)
    # Calcular el grado de depresión basado en la puntuación HAM-D
    if ham_d_score <= 7:
        depression_level = "Normal"
        background_color = "green"
        text_color = "white"
    elif ham_d_score <= 16:
        depression_level = "Leve"
        background_color = "lightgreen"
        text_color = "black"
    elif ham_d_score <= 23:
        depression_level = "Moderada"
        background_color = "yellow"
        text_color = "black"
    elif ham_d_score <= 30:
        depression_level = "Grave"
        background_color = "orange"
        text_color = "white"
    else:
        depression_level = "Muy grave"
        background_color = "red"
        text_color = "white"

    with col2:

        # Mostrar el grado de depresión
        st.subheader(f"El grado de depresión basado en la puntuación HAM-D es: {depression_level}")

        # Crear el mensaje con el color correspondiente al grado de depresión
        message = f"""
        <div style="width: 200px; height: 50px; margin: 0 auto; background-color: {background_color}; color: {text_color}; padding: 5px; border-radius: 5px; display: flex; align-items: center; justify-content: center;">
            <p style="font-size: 24px; font-weight: bold; margin: 0; text-align: center;">{depression_level}</p>
        </div>
        """

        # Mostrar el mensaje en el recuadro con el color correspondiente
        st.markdown(message, unsafe_allow_html=True)
        
        # depresion
        # Graficar el indicador de HAM-D
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ham_d_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Puntuación HAM-D", 'font': {'size': 24}},
            number={'font': {'size': 36}},
            gauge={
                'axis': {'range': [None, 52], 'tickwidth': 2, 'tickcolor': "black"},
                'bar': {'color': "white"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 7], 'color': "green"},
                    {'range': [7, 16], 'color': "lightgreen"},
                    {'range': [16, 23], 'color': "yellow"},
                    {'range': [23, 30], 'color': "orange"},
                    {'range': [30, 52], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': ham_d_score
                }
            }
        ))
        st.plotly_chart(fig)
    
    # ______________________________________________________________________________________________________________________________________________
    # ______________________________________________NUBE DE PALABRAS________________________________________________________________________________
    # ______________________________________________________________________________________________________________________________________________
    # Generar la nube de palabras a partir de las respuestas adicionales
    additional_texts = " ".join([response["text_response"] for response in st.session_state.responses if response["text_response"]])
    
    # Función para limpiar y procesar el texto
    def clean_text(text):
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Eliminar menciones
        text = re.sub(r'@\w+', '', text)
        
        # Eliminar hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Eliminar signos de puntuación
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenizar el texto
        tokens = nltk.word_tokenize(text)
        
        # Eliminar stopwords
        stop_words = set(stopwords.words('spanish'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Aplicar stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        
        # Unir los tokens de vuelta a un texto
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text

    # Limpiar el texto en additional_texts
    cleaned_text = clean_text(additional_texts)
    
    
    if cleaned_text:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
        
        st.subheader("Nube de palabras de la información adicional:")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        with st.container():
            st.pyplot(fig)
            
                # Inicializar el analizador de sentimiento
        sid = SentimentIntensityAnalyzer()

        # Analizar sentimiento para el texto completo
        #st.info(cleaned_text)
        sentiment = sid.polarity_scores(cleaned_text)

        # Calcular el promedio de los sentimientos
        avg_sentiment = {
            "neg": sentiment["neg"],
            "neu": sentiment["neu"],
            "pos": sentiment["pos"],
        }

        # Mostrar el análisis de sentimiento
        st.subheader("Análisis de sentimiento de la información adicional:")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"##### Score del análisis de sentimiento:")
            st.markdown(f'<div style="padding:10px;background-color:red;color:white;border-radius:5px;">Negativo: {avg_sentiment["neg"]:.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="padding:10px;background-color:gray;color:white;border-radius:5px;">Neutro: {avg_sentiment["neu"]:.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="padding:10px;background-color:green;color:white;border-radius:5px;">Positivo: {avg_sentiment["pos"]:.2f}</div>', unsafe_allow_html=True)

        with col2:
            # Crear una gráfica de barras con Plotly
            fig = go.Figure(data=[
                go.Bar(name='Negativo', x=['Sentimiento'], y=[avg_sentiment['neg']], marker_color='red'),
                go.Bar(name='Neutro', x=['Sentimiento'], y=[avg_sentiment['neu']], marker_color='gray'),
                go.Bar(name='Positivo', x=['Sentimiento'], y=[avg_sentiment['pos']], marker_color='green'),
            ])

            # Actualizar el diseño de la gráfica para una mejor visualización
            fig.update_layout(barmode='group', title="Resultados del Análisis de Sentimiento", xaxis_title="Tipo de Sentimiento", yaxis_title="Promedio")
            st.plotly_chart(fig, use_container_width=True)
            
    # Crear tres columnas para alinear el botón "Atrás" a la derecha
    _, _, col_atras = st.columns([1, 10, 1])
    
    with col_atras:
        if st.button("Atrás"):
            st.session_state.show_report = False
            st.experimental_rerun()
            
            
            
            
# Inicializar el estado de la sesión si no está definido
if 'password_correct' not in st.session_state:
    st.session_state.password_correct = False

if 'show_report' not in st.session_state:
    st.session_state.show_report = False

if 'show_questions' not in st.session_state:
    st.session_state.show_questions = False

if 'personal_info' not in st.session_state:
    st.session_state.personal_info = False

if not st.session_state.password_correct:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        show_password_prompt()
else:
    # Mostrar el formulario de información personal si no se ha completado
    if not st.session_state.personal_info:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            show_personal_info_form()
    # Mostrar el formulario de preguntas si se ha completado el formulario de información personal
    elif not st.session_state.show_report:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            show_form()
    # Mostrar el informe si se han completado ambos formularios
    else:
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            show_report()