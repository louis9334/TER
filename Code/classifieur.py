import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv('ww_game_data.csv', sep=';')


preprocessor = ColumnTransformer(
    transformers=[
        ('vect', CountVectorizer(), 'phrase'),  #Prétraitement pour les données textuelles
        ('nom_enc', OneHotEncoder(handle_unknown='ignore'), ['nom_joueur']),  #Encodage OneHot pour le nom du joueur
        ('role_enc', OneHotEncoder(handle_unknown='ignore'), ['role_joueur']),  #convertir des variables catégorielles
    ],
    remainder='drop' #Ignorer les autres colonnes
)

# Séparation des données en ensembles d'entraînement et de test
X = df.drop('mensonge', axis=1)
y = df['mensonge']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#init des modèles
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}


for name, model in models.items():
    #pour enchainer le pretraitement puis le classifieur
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    #Entraînement du modèle
    pipeline.fit(X_train, y_train)
    
    #Prédictions sur l'ensemble de test
    y_pred = pipeline.predict(X_test)
    
    # Calcul de l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Calcul et affichage de la matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"{name} Matrice de confusion :")
    print(conf_matrix)
    print("\n")
   
 display(

