import sklearn.metrics


def get_metrics_from_prediction(y, pred_proba):
    """
    Calcule les données de validation d'après les probabilités retournées par un modèle.
    :param y: Un tableau à une dimension contenant des labels
    :param pred_proba: Un tableau à une dimension contenant les probabilités retournées par un modèle
    :return: un dict contenant: auroc, score-f1, accuracy, sensibilité, spécificité
    """

    metrics = {}
    metrics["auroc"] = sklearn.metrics.roc_auc_score(y, pred_proba)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y, pred_proba > 0.5).ravel()
    metrics["f1"] = 2 * tp / (2 * tp + fp + fn)
    metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    metrics["sensitivity"] = tp / (tp + fn)
    metrics["specificity"] = tn / (tn + fp)

    return metrics