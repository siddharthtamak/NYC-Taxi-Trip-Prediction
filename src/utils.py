import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance 


def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def my_precision_score(y_true, y_pred, zero_division=0):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    denom = tp + fp
    if denom == 0:
        return float(zero_division)
    return tp / denom


def my_recall_score(y_true, y_pred, zero_division=0):
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = tp + fn
    if denom == 0:
        return float(zero_division)
    return tp / denom


def my_f1_score(y_true, y_pred, zero_division=0):
    p = my_precision_score(y_true, y_pred, zero_division=zero_division)
    r = my_recall_score(y_true, y_pred, zero_division=zero_division)
    denom = p + r
    if denom == 0:
        return float(zero_division)
    return 2 * (p * r) / denom


def evaluate_classifier(name, y_true, y_pred, y_proba=None, zero_division=0, show_report=True):

    # Convert and validate
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape. got {y_true.shape} vs {y_pred.shape}")

    # Convert probabilities if provided
    roc_auc = float('nan')
    if y_proba is not None:
        y_proba = np.asarray(y_proba).ravel()
        if y_proba.shape != y_true.shape:
            # try to handle shape (n_samples,1) or two-column probs
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                # assume second column is prob for class 1
                y_proba = y_proba[:, 1]
            else:
                raise ValueError("y_proba shape incompatible with y_true")
        # compute roc_auc safely
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            roc_auc = float('nan')

    # Basic metrics
    accuracy = float(np.mean(y_true == y_pred))
    precision = float(my_precision_score(y_true, y_pred, zero_division=zero_division))
    recall = float(my_recall_score(y_true, y_pred, zero_division=zero_division))
    f1 = float(my_f1_score(y_true, y_pred, zero_division=zero_division))
    cm = confusion_matrix(y_true, y_pred)

    # Print neatly
    print(f"--- {name} ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc if not np.isnan(roc_auc) else 'nan'}")
    print("Confusion Matrix:")
    print(cm)
    if show_report:
        try:
            print("Classification Report:")
            print(classification_report(y_true, y_pred, zero_division=zero_division))
        except Exception:
            pass
    print()

    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }