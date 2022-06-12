# Prédiction de la réponse postopératoire à la chirurgie du trou maculaire

## Modèle de régression logistique

### Description

Les données proviennent du travail de maîtrise de Mathieu Godbout, au département d'informatique de l'université Laval.

On peut télécharger les données au lien suivant:

https://www.kaggle.com/datasets/mathieugodbout/oct-postsurgery-visual-improvement

Le protocole employé est décrit dans la publication suivante:

https://tvst.arvojournals.org/article.aspx?articleid=2778731

On a donc utilisé une combinaison de l'ensemble de validation et de l'ensemble d'entraînement 
pour construire un modèle de régression logistique prédisant une réponse positive
à une intervention chirurgicale visant à réparer un trou maculaire.

Le modèle employé est fourni par la librairie scikit-learn:

LogisticRegressionCV

La variable dépendante est: une réponse positive à 6 mois post-op, définie
comme une amélioration de 15 lettres ou plus au score d'acuité visuelle (il s'agit
en fait du logMAR converti en ETDRS)

Les variables indépendantes:

- Présence d'un pseudocristallin
- Présence de rebords surélevés au pourtour du trou
- Diamètre du trou
- Durée de la maladie
- Acuité visuelle de base

### À faire

Selon le protocole décrit dans l'article il faudrait rebrasser et séparer
de nouveau les ensembles d'entraînement et de test afin de faire plusieurs expériences

Il conviendrait de vérifier si certaines variables indépendantes peuvent
être éliminées sans altérer la qualité de la prédiction
