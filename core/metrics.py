import sklearn.metrics


def get_metrics_from_prediction(y, pred_proba):
    metrics = {}

    metrics["auroc"] = sklearn.metrics.roc_auc_score(y, pred_proba)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y, pred_proba > 0.5).ravel()
    metrics["f1"] = 2 * tp / (2 * tp + fp + fn)
    metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    metrics["sensitivity"] = tp / (tp + fn)
    metrics["specificity"] = tn / (tn + fp)

    return metrics