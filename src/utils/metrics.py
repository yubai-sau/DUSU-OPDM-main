import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support

def compute_metrics(y_true, y_pred):
    oa = accuracy_score(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {'OA': oa, 'Kappa': kap, 'Precision': prec, 'Recall': rec, 'F1': f1}
