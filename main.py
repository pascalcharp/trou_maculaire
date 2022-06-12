import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import sklearn.metrics

from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing as prep

def pre_traitement_donnees (df):
    """
    Prétraitement d'un dataframe pour entraînement ou test.
    :param df: Dataframe provenant des données de https://www.kaggle.com/datasets/mathieugodbout/oct-postsurgery-visual-improvement lu
    avec la méthode read_csv
    :return: Un tuple (y, X) où y est un vecteur contenant la variable indépendante y (true si amélioration de l'acuité
    visuelle >= 15 lettres, false sinon) et X l'ensemble des variables dépendantes normalisées. (Présence d'un
    pseudocristallin, durée de la maladie, rebords élevés du trou maculaire, diamètre du trou maculaire, acuité visuelle de base)
    """

    # On tient seulement compte de 6 mois par-rapport au baseline
    df.drop(['id', 'age', 'sex', 'VA_2weeks', 'VA_3months', 'VA_12months'], inplace=True, axis=1)
    
    # On élimine toutes les rangées où il manques des variables
    df.dropna(inplace=True)

    # On redéfinit les index pour qu'il n'y ait pas de saut d'index

    df.reset_index(drop=True, inplace=True)
    
    # On définit une réponse positive comme une amélioration de >= 15 de l'acuité
    df['response'] = (df['VA_6months'] - df['VA_baseline']) >= 15
    
    # Variable dépendante: réponse positive ou non
    y = df.response.copy()
    
    # Variables indépendantes: pseudophakie, durée, rebord surélevé, grandeur du trou maculaire, acuité de base
    X = df.drop(['VA_6months', 'response'], axis=1)
    
    # On normalise les variables indépendantes continues à N(0,1)
    
    X_normal = X.copy()
    for variable in ['mh_duration', 'mh_size', 'VA_baseline']:
        X_normal[variable] = (X_normal[variable] - X_normal[variable].mean()) / X_normal[variable].std()

    return y, X_normal

#######################################################
# Lecture et prétraitement de l'ensemble d'entraînement
#######################################################

# Pour la régression logistique, on va combiner l'ensemble d'entraînement et l'ensemble de validation
df_train = pd.read_csv("~/PycharmProjects/trou_maculaire_regression_logistique/data/clinical_data_train.csv", na_values='-9')
df_val = pd.read_csv("~/PycharmProjects/trou_maculaire_regression_logistique/data/clinical_data_val.csv", na_values='-9')

df_train = pd.concat([df_train, df_val], ignore_index=True)

# On va formatter les séries pour l'entraînement
y_train, X_train = pre_traitement_donnees(df_train)

# On vérifie visuellement l'intégrité des données
print(y_train.to_string())
print(X_train.to_string())

################################################
# Modèle de régression et entraînement du modèle
################################################

model = LogisticRegressionCV(cv=10).fit(X_train, y_train)

################################################
# Lecture et prétraitement de l'ensemble de test
################################################

df_test = pd.read_csv("~/PycharmProjects/trou_maculaire_regression_logistique/data/clinical_data_test.csv", na_values='-9')
y_test, X_test = pre_traitement_donnees(df_test)

# On vérifie visuellement l'intégrité des données
print(y_test.to_string())
print(X_test.to_string())

##############################
# Test du modèle de régression
##############################

y_pred = pd.Series(model.predict(X_test), name='prediction')
compare = pd.concat([y_test, y_pred], axis=1)
print(compare.to_string())
print ('Exactitude: ', metrics.accuracy_score(y_test, y_pred))

##########################################
# Analyse de la courbe récepteur-opérateur
##########################################

scores = model.predict_proba(X_test)
print(scores)
faux_pos, vrais_pos, seuils = sklearn.metrics.roc_curve(y_true=y_test, y_score=scores[:, 1], pos_label=True)
auroc = sklearn.metrics.auc(faux_pos, vrais_pos)

print('AUROC = ', auroc)

########################
# Matrice de 'confusion'
#######################

print(sklearn.metrics.confusion_matrix(y_test, y_pred))

#########################
# Scores du modèle
#########################

print(sklearn.metrics.classification_report(y_test, y_pred))




