import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    # 1. Chargement des données Kaggle
    # Assure-toi que le fichier est bien dans data/train.csv
    df = pd.read_csv("data/train.csv")
    
    # Séparation Features (X) et Target (y)
    # 'price_range' est la colonne à prédire
    X = df.drop("price_range", axis=1)
    y = df["price_range"]
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Modèle
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # 4. Évaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Modèle Mobile Price entraîné. Accuracy : {acc:.2f}")
    
    # 5. Sauvegarde
    os.makedirs('app', exist_ok=True)
    joblib.dump(model, "app/model.pkl")
    print("Modèle sauvegardé dans app/model.pkl")

if __name__ == "__main__":
    train()