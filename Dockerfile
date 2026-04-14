# 1. Image de base légère avec Python
FROM python:3.9-slim

# 2. Définition du répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copie du fichier des dépendances en premier (optimisation du cache Docker)
COPY requirements.txt .

# 4. Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copie de tout le contenu du projet dans le conteneur
COPY . .

# 6. Exposition du port utilisé par FastAPI
EXPOSE 8000

# 7. Commande pour démarrer l'API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]