import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import sklearn.metrics

from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing as prep

class Clinical_model:
    """
    class Clinical_model:  Cette classe sert à analyser les données du projet de Mathieu Godbout sur les outcomes
    postopératoires de la chirurgie du trou maculaire.  Elle se charge de lire les données, les prétraiter, séparer aléatoirement
    un ensemble de données d'entraînement et de test, initialiser
    un modèle de régression logistique, entraîner le modèle avec les données d'entraînement et tester et valider le
    modèle avec les données de test.  On peut ensuite rediviser les données aléatoirement afin de recommencer les
    expériences d'entraînement et de test, dans le but de vérifier la fiabilité et reproducibilité du modèle.

    Attributs:
    ---------
    mode (string): Indique à l'instance si elle est prête à prétraiter les donner, entrainer le modèle ou tester le
    modèle.  Par-exemple, le modèle ne peut être testé avant d'avoir été entraîné...

    validation (dict): Dictionnaire contenant les résultats des tests de validation, pour une exécution donnée d'une
    séquence d'entraînement et test.

    stats (dict list): Liste des dictionnaires de validation pour toutes les exécutions consécutives des séquences d'entrainement
    et tests. Constitue l'output principal de la classe.

    df_total (DataFrame): Objet contenant l'entièreté des données cliniques traitées.  C'est à partir de cet objet qu'on
    divise aléatoirement un ensemble entrainement et un ensemble test.

    log_reg_model (LogisticRegressionCV): Modèle de régression logistique, fourni par sklearn

    df_train (DataFrame): Objet contenant les données d'entraînement pour une exécution donnée.

    df_test (DataFrame): Objet contenant les données de test pour une exécution donnée.

    y_train (Series): Objet contenant la variable dépendante de l'ensemble d'entraînement

    X_train (DataFrame): Objet contenannt les variables indépendantes de l'ensemble d'entraînement

    y_test (Series): Objet contenant la variable dépendante de l'ensemble de test

    X_test (DataFrame): Objet contenant les variables indépendantes de l'ensemble de test

    Méthodes
    --------

    __init__ (chemin): lit les fichiers de données dans le nom de chemin complet chemin/clinical_data_train.csv, et
    initialise les attributs afin de préparer la première séquence d'entraînement.

    initialiser_les_donnees(): Sépare les données en ensembles de test et d'entraînement.  Appelle la méthode de
    classe pre_traiter_les_donnees afin de terminer le prétraitement.

    entrainer_le_modele(): Permet le calcul des coefficients du modèle.

    tester_le_modele(): Applique le modèle aux données test.  Avec les résultats, calcule le score d'exactitude,
    l'aire sous la courbe de la courbe récepteur-opérateur, la matrice de confusion, et d'autres statistiques du modèle.
    Stocke les résultats dans une liste de dictionnaires.

    reinitialiser_le_modele(): Réalloue les ensembles d'entraînement et de test à partir des données totales.  Réinitialise
    les paramètres afin de pouvoir réentrainer et retester le modèle.

    pre_traiter_les_donnees(): Élimine les lignes comportant des données manquantes, élimine les colonnes non désirées,
    calcule la colonne correspondant à l'outcome primaire de l'étude.

    """

    @classmethod
    def pre_traiter_les_donnees (cls, df):
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

        # On élimine toutes les rangées où il manque des variables
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

    def __init__(self, chemin_donnees):
        """
        Prépare les données et le modèle pour la régression logistique:
        1. Lit les données dans les fichiers appropriés
        2. Préformatte les données afin d'être fournies au modèle de régression
        3. Initialise le modèle de régression
        :param chemin_donnees: string contenant le répertoire des fichiers devant être lus.
        (Doit finir par un /)
        """

        self.mode = "pretraitement"

        # Contiendra les résultats des tests pour l'exécution courante
        self.validation = {}

        #Servira à accumuler les résultats de tests pour des exécutions multiples
        self.stats = []

        # Lecture des fichiers et concaténation dans un unique dataframe: df_total
        self.df_total = pd.read_csv(chemin_donnees + "clinical_data_train.csv", na_values='-9')
        self.df_total = pd.concat([self.df_total, pd.read_csv(chemin_donnees + "clinical_data_val.csv", na_values='-9')], ignore_index=True)
        self.df_total = pd.concat([self.df_total, pd.read_csv(chemin_donnees + "clinical_data_test.csv", na_values='-9')], ignore_index=True)

        self.initialiser_les_donnees()

        # Configuration du modèle de régression logistique: log_reg_model
        self.log_reg_model = LogisticRegressionCV(cv=10)

        self.mode = "entrainement"

    def entrainer_le_modele(self):
        """
        Si les données ont été prétraitées, on entraîne le modèle.
        """

        if self.mode == "entrainement":
            self.log_reg_model.fit(self.X_train, self.y_train)
            self.mode = "validation"
        else:
            raise RuntimeError("Tentative d'entrainer un modèle dont les données ne sont pas traitées.")

    def tester_le_modele(self):
        """
        Une fois le modèle entraîné, on utilise les données de test pour extraire les scores de validation standard du
        modèle.
        :return:
        """

        if self.mode == "validation":

            # On génère les probabilités prédites à partir des données test
            y_pred = pd.Series(self.log_reg_model.predict(self.X_test))

            self.validation['exactitude'] = metrics.accuracy_score(self.y_test, y_pred)
            self.validation['scores'] = self.log_reg_model.predict_proba(self.X_test)
            self.validation['faux_pos'], self.validation['vrais_pos'], self.validation['seuils'] = sklearn.metrics.roc_curve(y_true=self.y_test, y_score=self.validation['scores'][:,1], pos_label=True)
            self.validation['auroc'] = sklearn.metrics.auc(self.validation['faux_pos'], self.validation['vrais_pos'])
            self.validation['confusion_matrix'] = sklearn.metrics.confusion_matrix(self.y_test, y_pred)
            self.validation['coefficients'] = self.log_reg_model.coef_
            self.stats.append(self.validation)

            # On est prêt à recommencer
            self.mode = "valide"
        else:
            raise RuntimeError("Le modèle n'a pas été entrainé.")

    def initialiser_les_donnees(self):
        """
        Séparation des données en ensembles de test et d'entrainement.  Prétraitement des données.
        """

        if self.mode == "pretraitement":
            # On sépare au hasard un frame d'entraînement et un frame de test
            self.df_train = self.df_total.copy(deep=True)
            self.df_test = self.df_train.sample(frac=0.2).reset_index(drop=True)

            # On va formatter les séries pour l'entraînement et les tests
            self.y_train, self.X_train = Clinical_model.pre_traiter_les_donnees(self.df_train)
            self.y_test, self.X_test = Clinical_model.pre_traiter_les_donnees(self.df_test)


        else:
            raise RuntimeError("Tentative de réinitialisation des données")

    def reinitialiser_le_modele (self):
        """
        Après une exécution d'entrainement et test, repréparer les données et le modèle afin de réentrainer et retester.
        """

        if self.mode == "valide":
           self.mode = "pretraitement"
           self.initialiser_les_donnees()
           self.log_reg_model = LogisticRegressionCV(cv=10)
           self.validation = {}
           self.mode = "entrainement"
        else:
            raise RuntimeError("Tentative de réinitialisation sans validation préablable")

    def lire_aurocs(self):
        """
        Retourne une liste des AUROCS calculés lors des différentes exécutions.
        :return: pd.Series liste des AUC pour chaque exécution du modèle
        """
        liste_aurocs = []
        for r in self.stats:
            liste_aurocs.append(r['auroc'])
        return pd.Series(liste_aurocs)

    def lire_coefficients(self):
        """
        Retourne la liste des coefficients de régression calculés à chaque exécution
        :return: pd.Dataframe liste des listes des coefficients pour chaque exécution
        """

        liste_coef = []
        for r in self.stats:
            liste_coef.append(r['coefficients'])
        return liste_coef

if __name__ == "__main__":

    mod = Clinical_model("/Users/pascal/Desktop/Python/trou_maculaire/data/")

    for i in range(20):
        print(f"Run # {i}", )
        mod.entrainer_le_modele()
        mod.tester_le_modele()
        mod.reinitialiser_le_modele()

    df_auroc = mod.lire_aurocs()
    print(df_auroc.to_string())
    print(df_auroc.describe().to_string())

    df_coefs = mod.lire_coefficients()
    print(df_coefs)
    #print(df_auroc.describe().to_string())










