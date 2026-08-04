import os
import numpy as np
import pandas as pd

# Fixer la graine aléatoire pour la reproductibilité
np.random.seed(42)

def generate_port_data(start_date="2020-01-01", periods=72):
    """
    Génère un dataset mensuel réaliste pour Marsa Maroc (Trafic & Finance)
    et un dataset d'escales navires (Fichiers d'attente GI/GI/c).
    """
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # =========================================================================
    # 1. DONNÉES MENSUELLES (Pour Module 1: Prévision IA & Module 3: Finance)
    # =========================================================================
    dates = pd.date_range(start=start_date, periods=periods, freq="MS")
    
    # A. Tendance (T) : Croissance annuelle moyenne ~4.5%
    trend = np.linspace(75000, 115000, periods)
    
    # B. Saisonnalité (S) : Pics d'activité en été/automne
    seasonality = 1 + 0.12 * np.sin(2 * np.pi * np.arange(periods) / 12)
    
    # C. Bruit / Résidus (R) : Loi de Student (queues épaisses pour chocs logistiques)
    noise = 1 + np.random.standard_t(df=5, size=periods) * 0.03
    
    # Volume de conteneurs (EVP)
    evp_volume = (trend * seasonality * noise).astype(int)
    
    # Escales de navires (corrélées au volume EVP)
    ship_counts = (evp_volume / np.random.uniform(600, 800, size=periods)).astype(int)
    
    # Variables Financières et Macroéconomiques
    # Prix des Bons du Trésor (Rf) fluctuant entre 3.2% et 4.1%
    rf_rate = np.random.uniform(0.032, 0.041, size=periods)
    
    # Cours de l'action Marsa Maroc (MSA) avec tendance haussière
    msa_stock = 220 + np.cumsum(np.random.normal(1.5, 5.0, size=periods))
    
    df_monthly = pd.DataFrame({
        "Date": dates,
        "Trafic_EVP": evp_volume,
        "Escales_Navires": ship_counts,
        "Taux_Bons_Tresor": np.round(rf_rate, 4),
        "Cours_MSA_MAD": np.round(msa_stock, 2)
    })
    
    df_monthly.to_csv("data/raw/port_monthly_data.csv", index=False)
    print("✅ Dataset Mensuel généré : 'data/raw/port_monthly_data.csv'")

    # =========================================================================
    # 2. DONNÉES D'ESCALES NAVIRES (Pour Module 2: Files d'attente GI/GI/c)
    # =========================================================================
    n_ships = 1000  # Échantillon de 1000 navires
    
    # Temps entre arrivées : Loi de Weibull (a=1.2, scale=35h) - Diallo et al. (2026)
    inter_arrival_hours = np.random.weibull(a=1.2, size=n_ships) * 35.0
    
    # Temps de service au quai : Loi Gamma (shape=2.5, scale=12h) - Diallo et al. (2026)
    service_hours = np.random.gamma(shape=2.5, scale=12.0, size=n_ships)
    
    df_ships = pd.DataFrame({
        "Ship_ID": [f"NAV-{i+1:04d}" for i in range(n_ships)],
        "Inter_Arrival_Hours": np.round(inter_arrival_hours, 2),
        "Service_Hours": np.round(service_hours, 2)
    })
    
    df_ships.to_csv("data/raw/ship_operations_data.csv", index=False)
    print("✅ Dataset Opérations Navires généré : 'data/raw/ship_operations_data.csv'")

if __name__ == "__main__":
    generate_port_data()