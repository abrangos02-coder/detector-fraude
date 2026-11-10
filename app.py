import os
import time
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import modelo as m

load_dotenv()
st.set_page_config(page_title="Detector de Fraude en Pagos en Linea", page_icon="🏦", layout="wide")

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
) if api_key else None


@st.cache_resource
def cargar_modelo():
    return m.entrenar()


modelo, test = cargar_modelo()

st.title("🏦 Detección de fraudes en pagos en línea")
st.caption("MVP: modelo de Machine Learning (Random Forest) + explicación con IA (Gemini)")

umbral = st.sidebar.slider("Umbral de alerta", 0.1, 0.9, 0.5, 0.05,
                           help="Mas bajo = detecta mas fraudes pero genera mas falsos positivos")
st.sidebar.write("Estado de la IA:", "✅ conectada" if client else "❌ sin API key")

tab1, tab2, tab3 = st.tabs(["🔍 Monitor en tiempo real", "🧪 Pruebas de estrés", "📊 Rendimiento del modelo"])

with tab1:
    st.subheader("Analizar una transacción")
    c1, c2, c3 = st.columns(3)
    with c1:
        monto = st.number_input("Monto (USD)", min_value=0.0, value=100.0)
        promedio = st.number_input("Gasto promedio del cliente (USD)", min_value=1.0, value=80.0)
        hora = st.slider("Hora", 0, 23, 14)
    with c2:
        pais_d = st.checkbox("País distinto al habitual")
        disp_n = st.checkbox("Dispositivo nuevo")
        tx_h = st.number_input("Transacciones en la última hora", 0, 50, 0)
    with c3:
        antig = st.number_input("Antigüedad de la cuenta (días)", 1, 5000, 900)

    if st.button("Analizar transacción", type="primary"):
        tx = dict(monto=monto, promedio=promedio, hora=hora, pais_distinto=int(pais_d),
                  dispositivo_nuevo=int(disp_n), tx_ultima_hora=tx_h, antiguedad_dias=antig)
        t0 = time.perf_counter()
        prob = m.puntaje(modelo, tx)
        latencia = (time.perf_counter() - t0) * 1000

        if prob >= umbral:
            st.error(f"🚨 SOSPECHOSA — probabilidad de fraude: {prob:.0%}")
        else:
            st.success(f"✅ LEGÍTIMA — probabilidad de fraude: {prob:.0%}")
        st.caption(f"Tiempo de respuesta del modelo: {latencia:.1f} ms")

        if client:
            prompt = f"""Eres analista de fraude bancario. Un modelo de ML calculo una probabilidad
de fraude de {prob:.0%} para esta transaccion:
- Monto: {monto} USD (promedio del cliente: {promedio} USD)
- Hora: {hora}:00 | Pais distinto: {pais_d} | Dispositivo nuevo: {disp_n}
- Transacciones en la ultima hora: {tx_h} | Antiguedad de la cuenta: {antig} dias
Explica en maximo 4 lineas, en espanol sencillo, por que es o no sospechosa y
recomienda una accion (aprobar, pedir verificacion de identidad o bloquear)."""
            
                        MODELOS = ["gemini-flash-lite-latest", "gemini-flash-latest",
                       "gemini-3.5-flash-lite", "gemini-3-flash-preview"]
            with st.spinner("La IA está redactando la explicación..."):
                texto, ultimo_error = None, None
                for nombre_modelo in MODELOS:
                    try:
                        r = client.chat.completions.create(
                            model=nombre_modelo,
                            messages=[{"role": "user", "content": prompt}])
                        texto = r.choices[0].message.content
                        break
                    except Exception as e:
                        ultimo_error = e
                if texto:
                    st.subheader("Explicación de la IA")
                    st.info(texto)
                else:
                    st.warning(f"No se pudo usar la IA: {ultimo_error}")
with tab2:
    st.subheader("Pruebas de estrés del modelo")
    st.write("Se evalúa si el modelo resiste intentos de engaño y fraudes nuevos.")
    if st.button("Ejecutar pruebas de estrés"):
        pruebas = {
            "Prueba 1: Datos manipulados a propósito (el defraudador oculta país, dispositivo y monto)":
                m.prueba_datos_manipulados(modelo, test, umbral),
            "Prueba 2: Nuevo tipo de fraude (muchas compras diminutas seguidas)":
                m.prueba_fraude_nuevo(modelo, test, umbral),
        }
        for nombre, res in pruebas.items():
            st.markdown(f"**{nombre}**")
            cols = st.columns(len(res))
            for col, (k, v) in zip(cols, res.items()):
                col.metric(k, v)
        st.markdown("**Prueba 3: Muchas transacciones al mismo tiempo**")
        vol = m.prueba_volumen(modelo, test)
        cols = st.columns(len(vol))
        for col, (k, v) in zip(cols, vol.items()):
            col.metric(k, v)
with tab3:
    st.subheader("Rendimiento sobre datos de prueba (separados del entrenamiento)")
    res = m.metricas(modelo, test, umbral)
    cols = st.columns(len(res))
    for col, (k, v) in zip(cols, res.items()):
        col.metric(k, v)
    st.write("**Variables que más influyen en la decisión del modelo:**")
    imp = pd.Series(modelo.feature_importances_, index=m.FEATURES).sort_values(ascending=False)
    st.bar_chart(imp)
    st.caption("Los datos son sintéticos (generados por código) con fines académicos.")
