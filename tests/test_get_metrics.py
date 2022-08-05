import sys
import numpy as np

sys.path.append(".")

from core import metrics

def test_get_metrics():
    y = np.array([0, 1, 0, 0, 1, 0, 1])
    p = np.array([0, 1, 0, 0, 1, 0, 1])

    res = {"auroc": 1.0, "accuracy": 1.0, "f1": 1.0, "sensitivity": 1.0, "specificity": 1.0}

    stats = metrics.get_metrics_from_prediction(y, p)
    assert(stats == res)

    y = np.array([0, 1, 0, 0, 1, 0, 1])
    p = np.array([1, 0, 1, 1, 0, 1, 0])

    res = {"auroc": 0.0, "accuracy": 0.0, "f1": 0.0, "sensitivity": 0.0, "specificity": 0.0}

    stats = metrics.get_metrics_from_prediction(y, p)
    assert (stats == res)

    y = np.array([0, 1, 0, 0, 1, 0, 1])
    p = np.array([0.4, 0.8, 0.4, 0.4, 0.8, 0.4, 0.8])

    res = {"auroc": 1.0, "accuracy": 1.0, "f1": 1.0, "sensitivity": 1.0, "specificity": 1.0}

    stats = metrics.get_metrics_from_prediction(y, p)
    assert(stats == res)

    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    p = np.array([0.6, 0.7, 0.8, 0.2, 0.3, 0.4, 0.8, 0.1])

    res = {"auroc": 0.8125, "accuracy": 0.75, "f1": 0.75, "sensitivity": 0.75, "specificity": 0.75}

    stats = metrics.get_metrics_from_prediction(y, p)
    assert(stats == res)


if __name__=="__main__":
    test_get_metrics()