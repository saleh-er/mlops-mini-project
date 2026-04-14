import streamlit as st
import joblib
import numpy as np
import os

# Configuration de la page
st.set_page_config(page_title="Mobile Price Predictor", layout="centered")

st.title("📱 Prédicteur de Gamme de Prix Mobile")
st.write("Ajustez les paramètres du téléphone pour obtenir une estimation du prix.")

# Chargement du modèle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Création de l'interface avec des colonnes
col1, col2 = st.columns(2)

with col1:
    ram = st.slider("RAM (MB)", 256, 4000, 2000)
    battery = st.slider("Puissance Batterie (mAh)", 500, 2000, 1200)
    int_memory = st.slider("Mémoire interne (GB)", 2, 64, 32)
    n_cores = st.selectbox("Nombre de cœurs", [1, 2, 3, 4, 5, 6, 7, 8])

with col2:
    px_height = st.number_input("Hauteur Écran (px)", 0, 2000, 500)
    px_width = st.number_input("Largeur Écran (px)", 0, 2000, 1000)
    mobile_wt = st.number_input("Poids du mobile (g)", 80, 200, 150)
    wifi = st.checkbox("Option WiFi")

# Préparation des données (on met des valeurs par défaut pour les colonnes non affichées)
if st.button("Estimer le prix"):
    # On crée un vecteur de 20 caractéristiques (ordre du dataset Kaggle)
    # Note: Pour simplifier ici, je ne mets que les variables clés, 
    # mais il faut remplir les 20 selon l'ordre de ton train.py
    features = np.zeros(20) 
    features[13] = ram
    features[0] = battery
    features[6] = int_memory
    features[9] = n_cores
    features[11] = px_height
    features[12] = px_width
    features[8] = mobile_wt
    features[19] = 1 if wifi else 0
    
    prediction = model.predict([features])[0]
    
    classes = ["Bas de gamme 📉", "Moyen de gamme 📊", "Haut de gamme 📈", "Luxe / Très haut de gamme 🔥"]
    st.success(f"Résultat : **{classes[prediction]}**")