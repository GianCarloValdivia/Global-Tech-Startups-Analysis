import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de página
st.set_page_config(page_title="Tech Startups Risk Predictor 2026", layout="wide", page_icon="🚀")

st.title("🚀 Global Tech Startups: Dashboard & Risk Predictor (2026)")
st.markdown("---")

# Cargar Modelo y Datos
@st.cache_data
def load_data():
    df = pd.read_csv("Data/processed/startups_cleaned_2026.csv")
    # Asegurar que exista la variable objetivo binaria Is_Closed
    if 'Is_Closed' not in df.columns:
        df['Is_Closed'] = np.where(df['Acquisition_Status'] == 'Closed', 1, 0)
    return df

@st.cache_resource
def load_model():
    return joblib.load("models/startup_risk_pipeline.pkl")

df = load_data()
model = load_model()

# Sidebar - Filtros
st.sidebar.header("🔍 Filtros de Exploración")
selected_domain = st.sidebar.multiselect("Selecciona Sector(es):", options=df['Domain'].unique(), default=list(df['Domain'].unique()[:3]))

filtered_df = df[df['Domain'].isin(selected_domain)] if selected_domain else df

# Métricas Principales
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Startups Analizadas", f"{len(filtered_df):,}")
col2.metric("Valoración Promedio", f"${filtered_df['Valuation_USD_Millions'].mean():.2f}M")
col3.metric("Burn Multiple Promedio", f"{filtered_df['Burn_Multiple'].mean():.2f}x")
col4.metric("Tasa de Cierre (Closed)", f"{(filtered_df['Is_Closed'].mean() * 100):.1f}%")

st.markdown("---")

# Sección de Predicción Interactiva
st.header("🤖 Simulador de Riesgo de Cierre para Nuevas Startups")

col_a, col_b = st.columns(2)

with col_a:
    domain = st.selectbox("Sector/Dominio:", df['Domain'].unique())
    city = st.selectbox("Ciudad:", df['City'].unique())
    country = st.selectbox("País:", df['Country'].unique())
    investor = st.selectbox("Nivel de Inversor:", df['Investor_Tier'].unique())
    ai_level = st.selectbox("Nivel de Adopción de IA:", ['High', 'Medium', 'Low'])
    stage = st.selectbox("Etapa de Financiamiento:", df['Funding_Stage'].unique())

with col_b:
    funding = st.number_input("Financiamiento Total (M$):", min_value=0.0, value=10.0)
    valuation = st.number_input("Valoración Actual (M$):", min_value=0.0, value=50.0)
    revenue = st.number_input("Revenue ARR (M$):", min_value=0.0, value=5.0)
    burn_rate = st.number_input("Burn Rate Mensual (M$):", min_value=0.0, value=0.5)
    runway = st.number_input("Meses de Runway:", min_value=0.0, value=12.0)

# Botón de Predicción
if st.button("Evaluar Riesgo de la Startup 🎯"):
    # Generar variables sintéticas
    text_profile = f"Startup in {domain} located in {city}, {country} backed by {investor} with {ai_level} AI adoption."
    burn_mult = round((burn_rate * 12) / revenue, 2) if revenue > 0 else 0.0

    input_data = pd.DataFrame([{
        'Total_Funding_USD_Millions': funding,
        'Valuation_USD_Millions': valuation,
        'Revenue_ARR_Millions': revenue,
        'Monthly_Burn_Rate_Millions': burn_rate,
        'Runway_Months_2024': runway,
        'Burn_Multiple': burn_mult,
        'Domain': domain,
        'Investor_Tier': investor,
        'AI_Adoption_Level': ai_level,
        'Funding_Stage': stage,
        'Text_Profile': text_profile
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ **Riesgo Alto de Cierre (Probabilidad: {probability:.1%})**")
    else:
        st.success(f"✅ **Startup Estable / Bajo Riesgo de Cierre (Probabilidad de fallo: {probability:.1%})**")