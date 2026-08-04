import numpy as np
import pandas as pd
from scipy.stats import weibull_min, gamma

class PortQueueSimulator:
    def __init__(self, berths_c=2):
        """
        :param berths_c: Nombre de postes à quai / portiques disponibles (c)
        """
        self.berths_c = berths_c
        
    def fit_distributions(self, df_ships):
        """
        Ajuste les lois de probabilité sur les données opérationnelles des navires :
        - Loi de Weibull pour les temps entre arrivées (Inter_Arrival_Hours)
        - Loi Gamma pour les temps de service à quai (Service_Hours)
        (Référence: Diallo et al., 2026)
        """
        # Ajustement Weibull (Arrivées)
        shape_w, loc_w, scale_w = weibull_min.fit(df_ships["Inter_Arrival_Hours"], floc=0)
        
        # Ajustement Gamma (Service)
        shape_g, loc_g, scale_g = gamma.fit(df_ships["Service_Hours"], floc=0)
        
        # Calcul des moyennes et coefficients de variation (CV)
        mean_inter_arrival = df_ships["Inter_Arrival_Hours"].mean()
        var_inter_arrival = df_ships["Inter_Arrival_Hours"].var()
        cv_a = np.sqrt(var_inter_arrival) / mean_inter_arrival
        
        mean_service = df_ships["Service_Hours"].mean()
        var_service = df_ships["Service_Hours"].var()
        cv_s = np.sqrt(var_service) / mean_service
        
        # Taux d'arrivée (lambda) et de service (mu) en navires/heure
        arrival_rate_lambda = 1.0 / mean_inter_arrival
        service_rate_mu = 1.0 / mean_service
        
        return {
            "lambda": arrival_rate_lambda,
            "mu": service_rate_mu,
            "cv_a": cv_a,
            "cv_s": cv_s,
            "mean_service_hours": mean_service
        }

    def calculate_queue_metrics(self, arrival_rate_lambda, service_rate_mu, cv_a, cv_s):
        """
        Calcule les métriques de la file d'attente GI/GI/c via l'approximation de Marchal / Allen-Cunneen.
        (Référence: Diallo et al., 2026)
        """
        c = self.berths_c
        # Taux d'occupation / utilisation des quais (rho)
        rho = arrival_rate_lambda / (c * service_rate_mu)
        
        if rho >= 1.0:
            # Système saturé : File d'attente infinie
            return {
                "occupancy_rate_rho": rho,
                "waiting_time_Wq_hours": float('inf'),
                "queue_length_Lq": float('inf'),
                "is_stable": False
            }
        
        # Approximation d'Allen-Cunneen pour le temps d'attente Wq dans GI/GI/c
        # Wq = [(cv_a^2 + cv_s^2) / 2] * [rho^sqrt(2(c+1)-1) / (c * (1 - rho))] * (1 / mu)
        traffic_variability = (cv_a**2 + cv_s**2) / 2.0
        exponent = np.sqrt(2 * (c + 1)) - 1
        wait_factor = (rho**exponent) / (c * (1.0 - rho))
        
        Wq_hours = traffic_variability * wait_factor * (1.0 / service_rate_mu)
        Lq = arrival_rate_lambda * Wq_hours  # Loi de Little (Lq = lambda * Wq)
        
        return {
            "occupancy_rate_rho": np.round(rho, 4),
            "waiting_time_Wq_hours": np.round(Wq_hours, 2),
            "queue_length_Lq": np.round(Lq, 2),
            "is_stable": True
        }

    def get_congestion_penalty(self, rho, base_tariff_mad=360.0):
        """
        Calcule la pénalité financière non linéaire due à la congestion D(rho).
        Plus le port approche de la saturation (rho -> 1), plus les retards imposent
        des rabais tarifaires ou des coûts d'attente pour conserver les armateurs.
        (Référence: Balliauw et al., 2019)
        """
        if rho <= 0.70:
            # Zone fluide : Aucune pénalité
            penalty_percentage = 0.0
        elif rho < 1.0:
            # Zone de friction : Pénalité exponentielle
            penalty_percentage = 0.15 * ((rho - 0.70) / 0.30)**2
        else:
            # Zone de saturation : Pénalité maximale de 30%
            penalty_percentage = 0.30
            
        adjusted_tariff = base_tariff_mad * (1.0 - penalty_percentage)
        return {
            "penalty_percentage": np.round(penalty_percentage * 100, 2),
            "adjusted_tariff_mad": np.round(adjusted_tariff, 2)
        }

# ==========================================
# Test Rapide du Module 2
# ==========================================
if __name__ == "__main__":
    # 1. Charger les données synthétiques d'escales navires
    df_ships = pd.read_csv("data/raw/ship_operations_data.csv")
    
    simulator = PortQueueSimulator(berths_c=2)
    
    # 2. Ajuster les distributions (Weibull & Gamma)
    params = simulator.fit_distributions(df_ships)
    print("📊 Paramètres extraits des données d'escales :")
    print(f" - Taux d'arrivée (λ) : {params['lambda']:.4f} navires/heure")
    print(f" - Taux de service (μ) : {params['mu']:.4f} navires/heure")
    print(f" - Coeff. Variation Arrivée (CV_a) : {params['cv_a']:.2f}")
    print(f" - Coeff. Variation Service (CV_s) : {params['cv_s']:.2f}")
    
    # 3. Calculer les métriques de file d'attente GI/GI/c
    metrics = simulator.calculate_queue_metrics(
        params['lambda'], params['mu'], params['cv_a'], params['cv_s']
    )
    print("\n⚓ Indicateurs de File d'Attente (GI/GI/c) :")
    print(f" - Taux d'occupation des quais (ρ) : {metrics['occupancy_rate_rho'] * 100}%")
    print(f" - Temps moyen d'attente au mouillage (Wq) : {metrics['waiting_time_Wq_hours']} heures")
    print(f" - Nombre moyen de navires en attente (Lq) : {metrics['queue_length_Lq']} navires")
    
    # 4. Calculer la pénalité financière de congestion
    penalty = simulator.get_congestion_penalty(metrics['occupancy_rate_rho'])
    print("\n💰 Impact Financier de la Congestion :")
    print(f" - Pénalité de tarif appliquée : -{penalty['penalty_percentage']}%")
    print(f" - Tarif effectif retenu : {penalty['adjusted_tariff_mad']} MAD/EVP")