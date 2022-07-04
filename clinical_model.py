import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import sklearn.metrics

from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing as prep
from sklearn.model_selection import KFold, StratifiedKFold


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

    folds (Array): Liste de groupes d'index définissant les 5 folds d'une exécution

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

    def __init__(self, chemin_donnees, experience="base"):
        """
        Prépare les données et le modèle pour la régression logistique:
        1. Lit les données dans les fichiers appropriés
        2. Préformatte les données afin d'être fournies au modèle de régression
        3. Initialise le modèle de régression
        :param experience:
        :param chemin_donnees: string contenant le répertoire des fichiers devant être lus.
        (Doit finir par un /)
        """

        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.y = None
        self.X = None

        self.mode = "pretraitement"

        #Servira à accumuler les résultats de tests pour les 5 runs
        self.stats = []

        # Les données seront groupées en 5 folds qui seront cross-validés tour à tour
        self.folds = None

        # Les folds seront générés par ceci
        self.df_groupes = None

        self.df_train_val = pd.read_csv(chemin_donnees + "clinical_data_train.csv", na_values='-9')
        self.df_train_val = pd.concat(
            [self.df_train_val, pd.read_csv(chemin_donnees + "clinical_data_val.csv", na_values='-9')],
            ignore_index=True)
        self.df_test = pd.read_csv(chemin_donnees + "clinical_data_test.csv", na_values='-9')
        self.df_total = pd.concat([self.df_train_val, self.df_test], ignore_index=True)

        if experience == "base":
            self.initialiser_les_donnees_pour_train_test()
        elif experience == "crossvalidation":
            self.initialiser_les_donnees_pour_crossval()
        else:
            raise RuntimeError("Type d'expérience non supporté")

        # Configuration du modèle de régression logistique: log_reg_model
        self.log_reg_model = LogisticRegressionCV(cv=10)

        self.mode = "entrainement"

    def cross_validation(self):
        """
        Si les données ont été prétraitées, on entraîne le modèle, ensuite on stocke les
        statistiques de tests dans la structure stats.
        """

        if self.mode == "entrainement":

            # Séparer les folds
            for train_index, test_index in self.df_groupes:
                self.X_train, self.X_test = self.X.iloc[train_index, :], self.X.iloc[test_index, :]
                self.y_train, self.y_test = self.y.iloc[train_index], self.y.iloc[test_index]

                # Entraîner le modèle de régression logistique
                self.log_reg_model.fit(self.X_train, self.y_train)
                self.mode = "validation"

                # Tester le modèle avec le fold réservé
                self.tester_le_modele()
        else:
            raise RuntimeError("Tentative d'entrainer un modèle dont les données ne sont pas traitées.")

    def entrainement_de_base(self):
        if self.mode == "entrainement":
            self.log_reg_model.fit(self.X_train, self.y_train)
            self.mode = "validation"
            self.tester_le_modele()



    def tester_le_modele(self):
        """
        Une fois le modèle entraîné, on utilise les données de test pour extraire les scores de validation standard du
        modèle.  Les résultats sont stockés dans l'objet stats.
        :return:
        """

        resultats_tests = {}

        if self.mode == "validation":

            # On génère les probabilités prédites à partir des données test
            y_pred = pd.Series(self.log_reg_model.predict(self.X_test))

            # Calculer les scores standards
            resultats_tests['exactitude'] = metrics.accuracy_score(self.y_test, y_pred)
            resultats_tests['scores'] = self.log_reg_model.predict_proba(self.X_test)
            resultats_tests['faux_pos'], resultats_tests['vrais_pos'], resultats_tests['seuils'] = sklearn.metrics.roc_curve(y_true=self.y_test, y_score= resultats_tests['scores'][:, 1], pos_label=True)
            resultats_tests['auroc'] = sklearn.metrics.auc(resultats_tests['faux_pos'], resultats_tests['vrais_pos'])
            resultats_tests['confusion_matrix'] = sklearn.metrics.confusion_matrix(self.y_test, y_pred)
            resultats_tests['coefficients'] = self.log_reg_model.coef_

            # Stocker les résultats dans l'objet stats
            self.stats.append(resultats_tests)

            # On est prêt à recommencer
            self.mode = "valide"
        else:
            raise RuntimeError("Le modèle n'a pas été entrainé.")

    def initialiser_les_donnees_pour_train_test(self):
        if self.mode == "pretraitement":
            self.y_train, self.X_train = Clinical_model.pre_traiter_les_donnees(self.df_train_val)
            self.y_test, self.X_test = Clinical_model.pre_traiter_les_donnees(self.df_test)
        else:
            raise RuntimeError("Mode invalide dans initialiser_les_donnees_pour_train_test")


    def initialiser_les_donnees_pour_crossval(self):
        """
        Prétraiter les données: extraire les colonnes pertinentes et calculer la variable dépendante.
        Séparer les variables dépendante et indépendante
        Séparation des données en 5 folds
        """

        if self.mode == "pretraitement":

            # On va séparer les variables dépendante et indépendante
            self.y, self.X = Clinical_model.pre_traiter_les_donnees(self.df_total)

            # Initialiser nos 5 folds
            self.folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
            self.df_groupes = self.folds.split(self.X, self.y)

        else:
            raise RuntimeError("Tentative de réinitialisation des données")

    def reinitialiser_le_modele(self, experience):
        """
        Après une exécution d'entrainement et test, repréparer les données et le modèle afin de réentrainer et retester.
        :param experience:
        """

        if self.mode == "valide":
            self.mode = "pretraitement"

            if experience == "crossvalidation":
                self.initialiser_les_donnees_pour_crossval()
            elif experience == "base":
                self.initialiser_les_donnees_pour_train_test()
            else:
                raise RuntimeError("Type d'expérience non supporté")


            self.log_reg_model = LogisticRegressionCV(cv=10)
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

    def afficher_courbe_ROC(self):

        plt.close()
        plt.figure()

        # Bâtir les courbes pour chaque exécution
        for i in range(len(self.stats)):
            results = self.stats[i]
            plt.plot(results["faux_pos"], results["vrais_pos"], color="grey", lw=1, label=f"Run {i}: AUROC = {results['auroc']:.2f}")

        fpr_moyen, tpr_moyen = self.construire_courbe_ROC_moyenne ()
        plt.plot(fpr_moyen, tpr_moyen, color="red", lw=2, linestyle = "dotted", label=f"Moyenne: AUROC = {self.lire_aurocs().mean()}")

        # Bâtir la courbe de référence
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

        # Paramètres du graphe
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Faux positifs")
        plt.ylabel("Vrais positifs")
        plt.title("Courbe récepteur-opérateur")
        plt.legend(loc="lower right")

        plt.show()

    def construire_courbe_ROC_moyenne(self):

        fpr_moyen = np.unique(np.concatenate([np.array(self.stats[i]["faux_pos"]) for i in range(5)]))
        tpr_moyen = np.zeros_like(fpr_moyen)

        for i in range(5):
            delta = np.interp(fpr_moyen, np.array(self.stats[i]["faux_pos"]), np.array(self.stats[i]["vrais_pos"]))
            tpr_moyen += delta

        tpr_moyen /= 5
        return fpr_moyen, tpr_moyen

    # (0, 0) (1, 2) (2, 6)
    # (0, 1) (2, 4) (3, 7)
    # (0, ) (1, ) (2, ) (3, )



if __name__ == "__main__":

    mod = Clinical_model("/Users/pascalcharpentier/PyCharmProjects/trou_maculaire_regression_logistique/data/", "crossvalidation")
    mod2 = Clinical_model("/Users/pascalcharpentier/PyCharmProjects/trou_maculaire_regression_logistique/data/", "base")

    mod.cross_validation()
    mod.afficher_courbe_ROC()

    mod2.entrainement_de_base()
    mod2.afficher_courbe_ROC()












