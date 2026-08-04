import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. Modèle LSTM en PyTorch pour la Tendance
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


# ==========================================
# 2. Classe Principale du Forecaster Hybride
# ==========================================
class STLHybridForecaster:
    def __init__(self, period=12, seq_length=6):
        """
        :param period: Période saisonnière (12 pour données mensuelles)
        :param seq_length: Taille de la fenêtre glissante pour le LSTM
        """
        self.period = period
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Modèles
        self.lstm_model = None
        self.arima_model = None
        self.xgb_model = None
        
        # Dataframes de décomposition
        self.trend = None
        self.seasonal = None
        self.resid = None

    def decompose(self, series):
        """Décomposition STL de la série temporelle"""
        stl = STL(series, period=self.period, robust=True)
        res = stl.fit()
        self.trend = res.trend
        self.seasonal = res.seasonal
        self.resid = res.resid
        return self.trend, self.seasonal, self.resid

    # --- Sous-module LSTM (Tendance) ---
    def _create_sequences(self, data):
        xs, ys = [], []
        for i in range(len(data) - self.seq_length):
            x = data[i:(i + self.seq_length)]
            y = data[i + self.seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def fit_lstm(self, trend_series, epochs=100, lr=0.01):
        scaled_trend = self.scaler.fit_transform(trend_series.values.reshape(-1, 1))
        X, y = self._create_sequences(scaled_trend)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        self.lstm_model = LSTMModel(input_size=1, hidden_layer_size=64, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=lr)
        
        self.lstm_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.lstm_model(X_tensor)
            loss = criterion(y_pred, y_tensor)
            loss.backward()
            optimizer.step()

    def predict_lstm(self, trend_series, steps):
        self.lstm_model.eval()
        scaled_trend = self.scaler.transform(trend_series.values.reshape(-1, 1))
        current_seq = scaled_trend[-self.seq_length:].tolist()
        
        predictions = []
        for _ in range(steps):
            x_tensor = torch.tensor([current_seq[-self.seq_length:]], dtype=torch.float32)
            with torch.no_grad():
                pred = self.lstm_model(x_tensor).item()
            predictions.append(pred)
            current_seq.append([pred])
            
        unscaled_preds = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return unscaled_preds.flatten()

    # --- Sous-module ARIMA (Saisonnalité) ---
    def fit_predict_arima(self, seasonal_series, steps):
        # Ajustement d'un ARIMA basique sur la composante saisonnière
        self.arima_model = ARIMA(seasonal_series, order=(2, 0, 2)).fit()
        forecast = self.arima_model.forecast(steps=steps)
        return forecast.values

    # --- Sous-module XGBoost (Résidus) ---
    def fit_predict_xgb(self, resid_series, steps):
        # Création de features de lags pour XGBoost
        df_xgb = pd.DataFrame({'Resid': resid_series})
        for lag in range(1, 4):
            df_xgb[f'lag_{lag}'] = df_xgb['Resid'].shift(lag)
        df_xgb.dropna(inplace=True)
        
        X = df_xgb.drop(columns=['Resid'])
        y = df_xgb['Resid']
        
        self.xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
        self.xgb_model.fit(X, y)
        
        # Prédiction itérative
        last_lags = list(resid_series.values[-3:])
        xgb_preds = []
        for _ in range(steps):
            features = np.array(last_lags[-3:][::-1]).reshape(1, -1)
            pred = self.xgb_model.predict(features)[0]
            xgb_preds.append(pred)
            last_lags.append(pred)
            
        return np.array(xgb_preds)

    # --- Entraînement et Prédiction globale ---
    def fit_predict(self, series, steps=12):
        """
        Entraîne le modèle hybride et prédit les 'steps' prochains mois.
        """
        print("[1/5] Decomposition STL de la serie...")
        self.decompose(series)
        
        print("[2/5] Entrainement du LSTM sur la Tendance...")
        self.fit_lstm(self.trend)
        trend_preds = self.predict_lstm(self.trend, steps)
        
        print("[3/5] Ajustement ARIMA sur la Saisonnalite...")
        seasonal_preds = self.fit_predict_arima(self.seasonal, steps)
        
        print("[4/5] Entrainement XGBoost sur les Residus...")
        resid_preds = self.fit_predict_xgb(self.resid, steps)
        
        print("[5/5] Recombinaison Multiplicative des predictions...")
        # Normalisation des composantes pour la combinaison multiplicative
        combined_forecast = trend_preds + seasonal_preds + resid_preds
        
        return pd.DataFrame({
            "Trend_Pred": trend_preds,
            "Seasonal_Pred": seasonal_preds,
            "Resid_Pred": resid_preds,
            "Combined_Forecast": np.round(combined_forecast).astype(int)
        })

# ==========================================
# 3. Test Rapide du Module
# ==========================================
if __name__ == "__main__":
    # Charger les données générées
    data_path = "data/raw/port_monthly_data.csv"
    df = pd.read_csv(data_path)
    
    traffic_series = pd.Series(df["Trafic_EVP"].values, index=pd.to_datetime(df["Date"]))
    
    forecaster = STLHybridForecaster(period=12)
    predictions = forecaster.fit_predict(traffic_series, steps=12)
    
    print("\n✅ Prévisions sur les 12 prochains mois :")
    print(predictions[["Combined_Forecast"]])