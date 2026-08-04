import os
import sys

# Ajouter le répertoire racine au chemin Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Importation des 4 modules cœurs du projet
from src.explainability import XAIExplainer
from src.finance import RealOptionsEvaluator
from src.forecasting import STLHybridForecaster
from src.queueing import PortQueueSimulator

# ==========================================
# 1. Configuration de la page Streamlit
# ==========================================
st.set_page_config(
    page_title="PortFinance-AI | Marsa Maroc",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS personnalisé
st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; color: #0E1117; font-weight: 700; text-align: center; }
    .sub-title { font-size: 1.1rem; color: #4A5568; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #F7FAFC; padding: 1rem; border-radius: 8px; border-left: 5px solid #3182CE; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='main-title'>⚓ PortFinance-AI : Aide à la Décision Stratégique</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-title'>Modélisation Prédicative, Théorie des Files d'Attente & Options Réelles pour Marsa Maroc</div>",
    unsafe_allow_html=True,
)

# ==========================================
# 2. Barre Latérale (Paramètres Globaux)
# ==========================================
st.sidebar.header("⚙️ Hypothèses & Paramètres")

st.sidebar.subheader("📊 Prévisions & Trafic")
forecast_horizon = st.sidebar.slider(
    "Horizon de prévision (Mois)", min_value=6, max_value=24, value=12
)

st.sidebar.subheader("⚓ Opérations Portuaires")
berths_c = st.sidebar.number_input(
    "Postes à quai disponibles (c)", min_value=1, max_value=5, value=2
)
base_tariff = st.sidebar.number_input(
    "Tarif de base (MAD/EVP)", min_value=100.0, max_value=1000.0, value=360.0
)

st.sidebar.subheader("💰 Finance & Risque (Marsa Maroc)")
wacc = st.sidebar.slider(
    "Coût Moyen Pondéré du Capital (WACC)",
    min_value=0.05,
    max_value=0.15,
    value=0.0809,
    step=0.005,
)
volatility = st.sidebar.slider(
    "Volatilité du sous-jacent (σ)",
    min_value=0.10,
    max_value=0.50,
    value=0.26,
    step=0.02,
)
n_simulations = st.sidebar.select_slider(
    "Simulations Monte-Carlo",
    options=[1000, 5000, 10000, 20000],
    value=5000,
)


# ==========================================
# 3. Chargement des Données Synthétiques
# ==========================================
@st.cache_data
def load_data():
    df_monthly = pd.read_csv("data/raw/port_monthly_data.csv")
    df_ships = pd.read_csv("data/raw/ship_operations_data.csv")
    return df_monthly, df_ships


try:
    df_monthly, df_ships = load_data()
except FileNotFoundError:
    st.error(
        "⚠️ Fichiers de données introuvables. Exécutez d'abord `python generate_synthetic_data.py` à la racine."
    )
    st.stop()

# ==========================================
# 4. Organisation par Onglets
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Module 1 : Prévision Hybride IA",
    "⚓ Module 2 : Congestion & Files GI/GI/c",
    "💎 Module 3 : Finance & Options Réelles",
    "🔍 Module 4 : Explicabilité (XAI)",
])

# -------------------------------------------------------------------------
# TAB 1 : PREVISION HYBRIDE (STL + LSTM + ARIMA + XGBoost)
# -------------------------------------------------------------------------
with tab1:
    st.header(
        "📈 Module 1 : Prévision de Trafic Portuaire (STL-LSTM-ARIMA-XGBoost)"
    )
    st.markdown("""
    Ce module décompose la série temporelle en **Tendance** (modélisée par **LSTM**), 
    **Saisonnalité** (modélisée par **ARIMA**), et **Résidus** (modélisés par **XGBoost**).
    """)

    if st.button("🚀 Lancer l'Entraînement & la Prévision", key="btn_forecast"):
        with st.spinner("Entraînement des modèles hybrides en cours..."):
            traffic_series = pd.Series(
                df_monthly["Trafic_EVP"].values,
                index=pd.to_datetime(df_monthly["Date"]),
            )
            forecaster = STLHybridForecaster(period=12)
            preds_df = forecaster.fit_predict(
                traffic_series, steps=forecast_horizon
            )

            # Sauvegarde des prédictions et du modèle XGBoost dans la session
            st.session_state["preds_df"] = preds_df
            st.session_state["forecaster"] = forecaster
            if hasattr(forecaster, "xgb_model"):
                st.session_state["xgb_model"] = forecaster.xgb_model

            st.success("✅ Prévisions générées avec succès !")

    if "preds_df" in st.session_state:
        preds_df = st.session_state["preds_df"]

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(
                df_monthly["Date"].tail(24),
                df_monthly["Trafic_EVP"].tail(24),
                label="Trafic Historique",
                color="#2B6CB0",
                linewidth=2,
            )

            future_dates = pd.date_range(
                start=pd.to_datetime(df_monthly["Date"].iloc[-1]),
                periods=forecast_horizon + 1,
                freq="MS",
            )[1:]
            ax.plot(
                future_dates.strftime("%Y-%m"),
                preds_df["Combined_Forecast"],
                label="Prévision Hybride",
                color="#DD6B20",
                linestyle="--",
                linewidth=2.5,
                marker="o",
            )

            plt.xticks(rotation=45)
            plt.ylabel("Volume Conteneurs (EVP)")
            plt.title("Trafic Mensuel Historique et Prédit")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)

        with col2:
            st.subheader("📊 Résultats Prédits")
            st.dataframe(
                preds_df[["Combined_Forecast"]].rename(
                    columns={"Combined_Forecast": "Trafic Prédit (EVP)"}
                )
            )

# -------------------------------------------------------------------------
# TAB 2 : CONGESTION ET FILES D'ATTENTE GI/GI/c
# -------------------------------------------------------------------------
with tab2:
    st.header(
        "⚓ Module 2 : Théorie des Files d'Attente GI/GI/c & Pénalités"
    )
    st.markdown("""
    Modélisation des escales de navires basée sur une **Loi de Weibull** (arrivées) 
    et une **Loi Gamma** (temps de service au quai).
    """)

    simulator = PortQueueSimulator(berths_c=berths_c)
    params = simulator.fit_distributions(df_ships)
    metrics = simulator.calculate_queue_metrics(
        params["lambda"], params["mu"], params["cv_a"], params["cv_s"]
    )
    penalty = simulator.get_congestion_penalty(
        metrics["occupancy_rate_rho"], base_tariff_mad=base_tariff
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Taux d'Occupation (ρ)", f"{metrics['occupancy_rate_rho'] * 100:.1f} %"
    )
    col2.metric(
        "Temps d'Attente Mouillage (Wq)",
        f"{metrics['waiting_time_Wq_hours']:.2f} hrs",
    )
    col3.metric("Navires en File (Lq)", f"{metrics['queue_length_Lq']:.2f}")
    col4.metric(
        "Tarif Effectif Ajusté", f"{penalty['adjusted_tariff_mad']:.1f} MAD"
    )

    st.divider()
    if penalty["penalty_percentage"] > 0:
        st.warning(
            f"⚠️ Alerte Congestion : Pénalité tarifaire de -{penalty['penalty_percentage']}% appliquée en raison du risque de saturation des quais."
        )
    else:
        st.success(
            "✅ Fluidité Portuaire : Le port fonctionne en zone fluide (Taux d'occupation < 70%)."
        )

# -------------------------------------------------------------------------
# TAB 3 : FINANCE & OPTIONS REELLES
# -------------------------------------------------------------------------
with tab3:
    st.header(
        "💎 Module 3 : Ingénierie Financière & Options Réelles (Call Extension)"
    )
    st.markdown("""
    Évaluation du projet d'extension par simulation **Monte-Carlo** (démonstration du biais de l'Inégalité de Jensen) 
    et valorisation de la flexibilité décisionnelle par **Black & Scholes**.
    """)

    evaluator = RealOptionsEvaluator(wacc=wacc, tax_rate=0.30)
    base_forecast_20y = np.linspace(75000, 150000, 20)

    if st.button(
        "🎲 Exécuter la Simulation Monte-Carlo & Options Réelles",
        key="btn_mc",
    ):
        with st.spinner(
            f"Exécution de {n_simulations} itérations stochastiques..."
        ):
            mc_results = evaluator.run_monte_carlo_van(
                base_forecast_20y,
                n_simulations=n_simulations,
                volatility=volatility,
            )
            st.session_state["mc_results"] = mc_results

    if "mc_results" in st.session_state:
        mc = st.session_state["mc_results"]
        call_val = evaluator.calculate_black_scholes_call(
            80.61e6, maturity_T=8, volatility=volatility
        )

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "VAN Statique Déterministe",
            f"{mc['van_deterministic_mad'] / 1e6:.2f} MDH",
        )
        col2.metric(
            "VAN Moyenne Monte-Carlo",
            f"{mc['van_monte_carlo_mean_mad'] / 1e6:.2f} MDH",
            delta=f"-{mc['jensen_bias_mad']/1e6:.2f} MDH (Biais Jensen)",
            delta_color="inverse",
        )
        col3.metric(
            "Valeur de l'Option Réelle (Call)", f"{call_val / 1e6:.2f} MDH"
        )

        st.divider()
        st.subheader(
            "🎯 Seuils Dynamiques de Déclenchement d'Extension (τ_t)"
        )
        thresholds = evaluator.calculate_dynamic_thresholds(
            base_forecast_20y, volatility=volatility
        )

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(
            range(1, 21),
            thresholds,
            color="#E53E3E",
            linestyle="-",
            marker="s",
            label="Seuil de Trafic Déclencheur (τ_t)",
        )
        ax.plot(
            range(1, 21),
            base_forecast_20y,
            color="#3182CE",
            linestyle="--",
            label="Trafic Prédit Moyen",
        )
        plt.xlabel("Années de Concession")
        plt.ylabel("Volume (EVP / Unités)")
        plt.title("Seuils d'Exercice Dynamiques de l'Option d'Extension")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

# -------------------------------------------------------------------------
# TAB 4 : EXPLICABILITE (XAI)
# -------------------------------------------------------------------------
with tab4:
    st.header(
        "🔍 Module 4 : IA Explicable (XAI) - Transparence Décisionnelle"
    )
    st.markdown("""
    Utilisation des valeurs **SHAP** et des **Partial Dependence Plots (PDP)** 
    pour auditer les facteurs clés influençant les prédictions du modèle.
    """)

    df_features = pd.DataFrame({
        "Trafic_Lag1": df_monthly["Trafic_EVP"].shift(1),
        "Trafic_Lag2": df_monthly["Trafic_EVP"].shift(2),
        "Escales_Navires": df_monthly["Escales_Navires"],
        "Taux_Bons_Tresor": df_monthly["Taux_Bons_Tresor"],
    }).dropna()

    y_target = df_monthly["Trafic_EVP"].iloc[2:]

    # Récupération ou entraînement à la volée du modèle XGBoost
    import xgboost as xgb

    if "xai_xgb_model" in st.session_state:
        model_xgb = st.session_state["xai_xgb_model"]
    else:
        # Entraînement du modèle XGBoost dédié aux features globales pour l'onglet XAI
        model_xgb = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        )
        model_xgb.fit(df_features, y_target)
        st.session_state["xai_xgb_model"] = model_xgb

    # Initialisation et exécution de l'expliqueur
    explainer = XAIExplainer(model=model_xgb)
    importance_df = explainer.get_feature_importance(df_features)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔍 Ranking des Prédicteurs (Importance SHAP)")
        st.dataframe(importance_df, use_container_width=True)

    with col2:
        st.subheader("📈 Partial Dependence Plot (PDP)")
        st.markdown(
            "Élasticité du trafic prédit face aux volumes du mois précédent (`Trafic_Lag1`) :"
        )

        x_vals = np.linspace(
            df_features["Trafic_Lag1"].min(),
            df_features["Trafic_Lag1"].max(),
            30,
        )
        pdp_vals = [
            np.mean(
                explainer.model.predict(
                    df_features.assign(Trafic_Lag1=val)
                )
            )
            for val in x_vals
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_vals, pdp_vals, color="#319795", linewidth=2.5)
        ax.set_xlabel("Volume Trafic Lag 1 (EVP)")
        ax.set_ylabel("Impact Moyen Prédit (EVP)")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)