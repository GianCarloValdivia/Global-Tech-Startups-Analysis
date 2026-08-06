# 🚀 Global Tech Startups Analysis & Risk Prediction (2026)

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/)

Un proyecto end-to-end de **Análisis Exploratorio de Datos (EDA)**, **Ingeniería de Características Vectorizadas**, **Procesamiento de Lenguaje Natural (NLP)** y **Pipelines de Machine Learning** para analizar y predecir el riesgo de cierre en 25,000 tech startups a nivel global.

---

## 📌 Tabla de Contenidos
1. [Descripción del Problema](#-descripción-del-problema)
2. [Estructura del Repositorio](#-estructura-del-repositorio)
3. [Análisis Exploratorio y NumPy (EDA)](#-análisis-exploratorio-y-numpy-eda)
4. [Machine Learning & Pipelines](#-machine-learning--pipelines)
5. [Instalación y Uso](#-instalación-y-uso)
6. [Autor](#-autor)

---

## 🎯 Descripción del Problema

El ecosistema global de startups tecnológicas enfrenta importantes desafíos financieros, altos niveles de tasa de quemado de capital (*burn rate*) y reestructuraciones de personal (*layoffs*). 

Este proyecto utiliza un dataset de **25,000 startups** para:
* Evaluar la salud financiera mediante **cálculos matriciales/vectoriales con NumPy**.
* Imputar y limpiar valores faltantes en niveles de adopción de Inteligencia Artificial mediante métodos no paramétricos.
* Extraer embeddings textuales mediante **TF-IDF Vectorizer** combinando atributos de dominio, ubicación e inversores.
* Construir un **Pipeline robusto de Scikit-Learn** para clasificar el riesgo de cierre (*Closed*) de una startup.
* Agrupar a las empresas en arquetipos de riesgo mediante **K-Means Clustering**.

---

## 📁 Estructura del Repositorio

```text
Global-Tech-Startups-Analysis/
├── Data/
│   ├── raw/                      # Dataset original (25,000 startups)
│   └── processed/                # Dataset limpio y transformado (CSV)
├── Images/                       # Gráficos exportados para reporte
│   ├── eda_summary_dashboard.png
│   └── model_evaluation_metrics.png
├── models/                       # Pipeline entrenado exportado (.pkl)
│   └── startup_risk_pipeline.pkl
├── Notebooks/
│   ├── 01_Analisis_Exploratorio.ipynb
│   └── 02_Machine_Learning_Pipelines.ipynb
├── .gitignore
├── README.md
└── requirements.txt