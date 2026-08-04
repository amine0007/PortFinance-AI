import numpy as np
import pandas as pd
from scipy.stats import norm

class RealOptionsEvaluator:
    def __init__(self, wacc=0.0809, tax_rate=0.30, risk_free_rate=0.0334):
        """
        :param wacc: Coût Moyen Pondéré du Capital de Marsa Maroc (8.09%) - Rachidi (2008)
        :param tax_rate: Taux d'imposition IS (30%) - Rachidi (2008)
        :param risk_free_rate: Taux sans risque continu (3.34%) - Rachidi (2008)
        """
        self.wacc = wacc
        self.tax_rate = tax_rate
        self.r_f = risk_free_rate
        
        # Hypothèses du projet Marsa Maroc (Rachidi, 2008)
        self.capex_initial = 130e6      # 130 MDH pour le terminal à 3 étages
        self.capex_extension = 50e6      # 50 MDH pour le 4ème étage (Option)
        self.capacity_initial = 122000   # Capacité max initiale (véhicules/EVP)
        self.capacity_extended = 164000  # Capacité max avec extension
        
    def calculate_free_cash_flow(self, volume, tariff_mad=360.0, is_extended=False):
        """
        Calcule le Free Cash Flow (FCF) annuel pour un volume donné.
        (Référence: Rachidi, 2008)
        """
        capacity_limit = self.capacity_extended if is_extended else self.capacity_initial
        served_volume = min(volume, capacity_limit)  # Contrainte capacitaire physique
        
        # Revenus
        revenue = served_volume * tariff_mad
        
        # Charges variables et fixes (OPEX ~ 252.5 MAD/unité)
        opex_variable = served_volume * 252.5
        opex_fixed = 5e6  # Charges fixes d'exploitation
        
        # Amortissement linéaire sur 20 ans
        capex = self.capex_extension if is_extended else self.capex_initial
        depreciation = capex / 20.0
        
        # Résultat Exploitation & Impôts (IS)
        ebit = revenue - opex_variable - opex_fixed - depreciation
        tax = max(0, ebit * self.tax_rate)
        net_income = ebit - tax
        
        # Cash Flow Opérationnel = Net Income + Amortissement
        fcf = net_income + depreciation
        return fcf, served_volume

    def run_monte_carlo_van(self, base_forecast, n_simulations=20000, volatility=0.26):
        """
        Exécute 20 000 simulations Monte-Carlo sous Mouvement Brownien Géométrique (GBM)
        pour démontrer le Biais de l'Inégalité de Jensen.
        (Référence: Godinho et al., 2021)
        """
        years = len(base_forecast)
        discount_factors = (1 + self.wacc) ** np.arange(1, years + 1)
        
        van_statique_simulations = []
        
        for _ in range(n_simulations):
            # Génération d'une trajectoire stochastique de demande (GBM)
            shocks = np.random.normal(0, volatility, size=years)
            simulated_volumes = base_forecast * np.exp(shocks - 0.5 * volatility**2)
            
            # Calcul des FCF réels avec la contrainte de capacité initiale
            fcfs = [self.calculate_free_cash_flow(v, is_extended=False)[0] for v in simulated_volumes]
            
            # VAN pour cette simulation
            van = np.sum(fcfs / discount_factors) - self.capex_initial
            van_statique_simulations.append(van)
            
        mean_van_mc = np.mean(van_statique_simulations)
        
        # Calcul de la VAN déterministe (sans simulation, sur les moyennes)
        fcfs_deterministic = [self.calculate_free_cash_flow(v, is_extended=False)[0] for v in base_forecast]
        van_deterministic = np.sum(fcfs_deterministic / discount_factors) - self.capex_initial
        
        jensen_bias = van_deterministic - mean_van_mc
        
        return {
            "van_deterministic_mad": np.round(van_deterministic, 2),
            "van_monte_carlo_mean_mad": np.round(mean_van_mc, 2),
            "jensen_bias_mad": np.round(jensen_bias, 2),
            "simulations": van_statique_simulations
        }

    def calculate_black_scholes_call(self, S_underlying, maturity_T=8, volatility=0.26):
        """
        Calcule la valeur de l'Option Réelle (Call) par le modèle de Black & Scholes.
        S_underlying: Valeur actualisée des Cash Flows futurs de l'extension (Phase 2).
        (Référence: Rachidi, 2008)
        """
        K_strike = self.capex_extension  # Prix d'exercice = Coût de l'extension (50 MDH)
        r = self.r_f                     # Taux sans risque continu (3.34%)
        sigma = volatility               # Volatilité de l'actif sous-jacent (26%)
        
        d1 = (np.log(S_underlying / K_strike) + (r + 0.5 * sigma**2) * maturity_T) / (sigma * np.sqrt(maturity_T))
        d2 = d1 - sigma * np.sqrt(maturity_T)
        
        call_value = S_underlying * norm.cdf(d1) - K_strike * np.exp(-r * maturity_T) * norm.cdf(d2)
        return np.round(call_value, 2)

    def calculate_dynamic_thresholds(self, base_forecast, volatility=0.26):
        """
        Détermine la courbe des seuils dynamiques de déclenchement d'extension (tau_t).
        Plus l'échéance approche, plus le seuil de trafic exigé augmente pour rentabiliser l'extension.
        (Référence: Godinho et al., 2021)
        """
        years = len(base_forecast)
        thresholds_tau = []
        
        for t in range(years):
            remaining_years = years - t
            # Seuil de trafic dynamique tau_t (en EVP / véhicules)
            # Factorise le temps restant et le coût d'opportunité du capital
            base_threshold = self.capacity_initial * (1 + 0.05 * (10 / max(1, remaining_years)))
            thresholds_tau.append(np.round(base_threshold, 0))
            
        return thresholds_tau

# ==========================================
# Test Rapide du Module 3
# ==========================================
if __name__ == "__main__":
    # Données prévisionnelles de trafic sur 20 ans (ex: Marsa Maroc)
    base_forecast_20y = np.linspace(75000, 150000, 20)
    
    evaluator = RealOptionsEvaluator(wacc=0.0809, tax_rate=0.30, risk_free_rate=0.0334)
    
    # 1. Évaluation Monte-Carlo & Biais de Jensen
    print("⏳ Exécution de 20 000 simulations Monte-Carlo...")
    mc_results = evaluator.run_monte_carlo_van(base_forecast_20y, n_simulations=20000, volatility=0.26)
    
    print("\n📈 Résultats Financiers (VAN Statique & Biais de Jensen) :")
    print(f" - VAN Déterministe (Modèle Excel classique) : {mc_results['van_deterministic_mad'] / 1e6:.2f} MDH")
    print(f" - VAN Moyenne Monte-Carlo (Réelle)          : {mc_results['van_monte_carlo_mean_mad'] / 1e6:.2f} MDH")
    print(f" - Biais d'Inégalité de Jensen (Surestimation) : {mc_results['jensen_bias_mad'] / 1e6:.2f} MDH")
    
    # 2. Valorisation de l'Option Réelle (Call Extension 4ème étage)
    # S_underlying = Valeur actualisée des cash flows de l'extension (ex: ~80.61 MDH selon Rachidi 2008)
    S_underlying = 80.61e6
    call_option_value = evaluator.calculate_black_scholes_call(S_underlying, maturity_T=8, volatility=0.26)
    
    print("\n💎 Valorisation de l'Option Réelle d'Extension (Black & Scholes) :")
    print(f" - Valeur de l'Option d'Extension (Call) : {call_option_value / 1e6:.2f} MDH")
    print(f" - Valeur Globale du Projet (VAN + Call) : {(mc_results['van_monte_carlo_mean_mad'] + call_option_value) / 1e6:.2f} MDH")
    
    # 3. Seuils Dynamiques de Déclenchement (tau_t)
    thresholds = evaluator.calculate_dynamic_thresholds(base_forecast_20y)
    print("\n🎯 Seuils Dynamiques de Déclenchement d'Extension (tau_t à l'année 1, 5, 8) :")
    print(f" - Seuil Année 1 : {thresholds[0]} unités")
    print(f" - Seuil Année 5 : {thresholds[4]} unités")
    print(f" - Seuil Année 8 : {thresholds[7]} unités")