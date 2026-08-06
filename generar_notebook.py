import json
import os

# Estructura del notebook con todas las celdas de análisis
notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Análisis Exploratorio de Datos (EDA) - Global Tech Startups\n",
                "**Módulos 1 y 2:** Introducción a Python/NumPy y Análisis Exploratorio de Datos."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import os\n",
                "\n",
                "# 1. Cargar Dataset\n",
                "df = pd.read_csv('../Data/global_tech_startups_2026.csv')\n",
                "print(f'Dimensión del dataset: {df.shape}')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Operaciones Vectorizadas con NumPy (Módulo 1)\n",
                "Cálculo de métricas financieras vectorizadas."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Extracción de arrays NumPy\n",
                "valuations = df['Valuation_USD_Millions'].to_numpy()\n",
                "revenues = df['Revenue_ARR_Millions'].to_numpy()\n",
                "burn_rate = df['Monthly_Burn_Rate_Millions'].to_numpy()\n",
                "layoffs = df['Layoffs_2024_2025'].to_numpy()\n",
                "peak_headcount = df['Peak_Headcount_2023'].to_numpy()\n",
                "runway = df['Runway_Months_2024'].to_numpy()\n",
                "\n",
                "# Múltiplo de Valoración\n",
                "multiples = np.divide(valuations, revenues, out=np.zeros_like(valuations), where=revenues != 0)\n",
                "df['Valuation_Multiple'] = np.round(multiples, 2)\n",
                "\n",
                "# Burn Multiple\n",
                "burn_multiple = np.divide(burn_rate * 12, revenues, out=np.zeros_like(burn_rate), where=revenues > 0)\n",
                "df['Burn_Multiple'] = np.round(burn_multiple, 2)\n",
                "\n",
                "# Categorización de Salud de Runway\n",
                "conditions = [(runway < 6), (runway >= 6) & (runway <= 18), (runway > 18)]\n",
                "choices = ['Crítico (< 6m)', 'Moderado (6-18m)', 'Saludable (> 18m)']\n",
                "df['Runway_Health'] = np.select(conditions, choices, default='Desconocido')\n",
                "\n",
                "df[['Domain', 'Valuation_Multiple', 'Burn_Multiple', 'Runway_Health']].head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Tratamiento de Datos Faltantes e Imputación (Módulo 2)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Imputación no paramétrica por moda según el Sector (Domain)\n",
                "df['AI_Adoption_Level'] = df.groupby('Domain')['AI_Adoption_Level'].transform(\n",
                "    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Low')\n",
                ")\n",
                "\n",
                "print(f'Nulos restantes en AI_Adoption_Level: {df[\"AI_Adoption_Level\"].isnull().sum()}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Guardar Dataset Procesado\n",
                "os.makedirs('../Data/processed', exist_ok=True)\n",
                "df.to_csv('../Data/processed/startups_cleaned_2026.csv', index=False)\n",
                "print('✅ Dataset procesado guardado con éxito.')"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Asegurar carpeta de destino
os.makedirs("Notebooks", exist_ok=True)

# Escribir el archivo .ipynb válido
with open("Notebooks/01_Analisis_Exploratorio.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print("🎉 ¡Notebook '01_Analisis_Exploratorio.ipynb' reescrito con éxito!")