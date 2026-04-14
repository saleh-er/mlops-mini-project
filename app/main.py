from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# 1. Définition du schéma de données (les colonnes du dataset Mobile Price)
class MobileFeatures(BaseModel):
    battery_power: int
    blue: int
    clock_speed: float
    dual_sim: int
    fc: int
    four_g: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    sc_h: int
    sc_w: int
    talk_time: int
    three_g: int
    touch_screen: int
    wifi: int

# 2. Initialisation de l'API
app = FastAPI(title="Mobile Price Predictor API")

# 3. Chargement du modèle au démarrage
# Note : Assure-tu d'avoir exécuté train.py avant pour générer ce fichier !
model = joblib.load("app/model.pkl")

@app.get("/health")
def health():
    """Endpoint de santé pour vérifier si l'API est en ligne."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(features: MobileFeatures):
    """Endpoint pour prédire la gamme de prix (0, 1, 2 ou 3)."""
    # Conversion des données reçues en tableau NumPy pour le modèle
    data = np.array([[ 
        features.battery_power, features.blue, features.clock_speed, features.dual_sim,
        features.fc, features.four_g, features.int_memory, features.m_dep,
        features.mobile_wt, features.n_cores, features.pc, features.px_height,
        features.px_width, features.ram, features.sc_h, features.sc_w,
        features.talk_time, features.three_g, features.touch_screen, features.wifi
    ]])
    
    prediction = model.predict(data)
    
    return {
        "price_range_prediction": int(prediction[0]),
        "description": "0: Low, 1: Medium, 2: High, 3: Very High"
    }