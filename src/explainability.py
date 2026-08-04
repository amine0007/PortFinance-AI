import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


class XAIExplainer:
    """Module d'Explicabilité (XAI) pour le modèle XGBoost de PortFinance-AI."""

    def __init__(self, model=None):
        self.model = model
        self.explainer = None
        self.shap_values = None

    def fit_explainer(self, X_train, model=None):
        """Initialise le TreeExplainer en corrigeant la configuration JSON de XGBoost."""
        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("Un modèle doit être fourni à l'initialisation ou à fit_explainer().")

        # --- CORRECTIF ULTIME XGBOOST 2.x / SHAP ---
        model_to_explain = self.model
        if hasattr(self.model, "get_booster"):
            import json
            try:
                booster = self.model.get_booster()
                
                # 1. Extraction de la configuration interne au format JSON
                config = json.loads(booster.save_config())
                
                # 2. Navigation vers le paramètre base_score
                learner_params = config.get("learner", {}).get("learner_model_param", {})
                base_score = learner_params.get("base_score")
                
                # 3. Nettoyage si présence de crochets (ex: '[9.485086E4]')
                if isinstance(base_score, str) and base_score.startswith("[") and base_score.endswith("]"):
                    clean_score = base_score.strip("[]")
                    config["learner"]["learner_model_param"]["base_score"] = clean_score
                    booster.load_config(json.dumps(config))

                # 4. Expliquer le Booster directement pour éviter les wrappers incompatibles
                model_to_explain = booster
            except Exception:
                model_to_explain = self.model

        # Initialisation sécurisée de TreeExplainer avec le modèle ou son Booster nettoyé
        self.explainer = shap.TreeExplainer(model_to_explain)

    def calculate_shap_values(self, X):
        """Calcule les valeurs SHAP."""
        if self.explainer is None:
            self.fit_explainer(X)
        self.shap_values = self.explainer(X)
        return self.shap_values

    def get_feature_importance(self, X):
        """Calcule l'importance globale des variables via SHAP."""
        if self.explainer is None:
            self.fit_explainer(X)

        raw_shap = self.explainer.shap_values(X)
        if isinstance(raw_shap, list):
            raw_shap = raw_shap[0]

        mean_abs_shap = np.abs(raw_shap).mean(axis=0)

        importance_df = pd.DataFrame(
            {"Feature": X.columns, "Importance": mean_abs_shap}
        ).sort_values(by="Importance", ascending=False)

        return importance_df

    def plot_summary(self, X, max_display=10):
        """Génère le graphique SHAP Summary Plot."""
        if self.shap_values is None:
            self.calculate_shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(self.shap_values, X, max_display=max_display, show=False)
        plt.tight_layout()
        return fig