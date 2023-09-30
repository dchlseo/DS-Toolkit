# Data wrangling
import pandas as pd
import numpy as np

# Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.metrics import roc_curve

# Visualization
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------

# 1. GET METRICS

# 1-1. Return various performance metrics
def get_clf_eval(y_test, pred):
    """
    Calculate and return accuracy, precision, recall, F1 score, and AUC.

    Args:
        y_test (array-like): True labels.
        pred (array-like): Predicted labels.

    Returns:
        tuple: A tuple containing accuracy, precision, recall, F1 score, and AUC.
    """
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    re = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred)

    return acc, pre, re, f1, auc

# 1-2. Return confusion matrix and performance metrics
def print_clf_eval(y_test, pred):
    """
    Print confusion matrix, accuracy, precision, recall, F1 score, and AUC.

    Args:
        y_test (array-like): True labels.
        pred (array-like): Predicted labels.
    """
    confusion = confusion_matrix(y_test, pred)
    acc, pre, re, f1, auc = get_clf_eval(y_test, pred)

    print('=> Confusion matrix:')
    print(confusion)
    print('========')

    print('Accuracy: {0:.4f}, Precision: {1:.4f}'.format(acc, pre))
    print('Recall: {0:.4f}, F1 Score: {1:.4f}, AUC: {2:.4f}'.format(re, f1, auc))


# 2. MULTIPLE MODEL COMPARISONS

# 2-1. Return performance metrics for a single model.
def get_result(model, X_train, y_train, X_test, y_test):
    """
    Fit a model and return its performance metrics.

    Args:
        model: A machine learning model.
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        X_test (array-like): Test data features.
        y_test (array-like): Test data labels.

    Returns:
        tuple: A tuple containing accuracy, precision, recall, F1 score, and AUC.
    """
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return get_clf_eval(y_test, pred)

# 2-2. Return performance metrics for multiple models
def get_result_pd(models, model_names, X_train, y_train, X_test, y_test):
    """
    Fit multiple models and return their performance metrics as a DataFrame.

    Args:
        models (list): List of machine learning models.
        model_names (list): List of model names.
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        X_test (array-like): Test data features.
        y_test (array-like): Test data labels.

    Returns:
        pd.DataFrame: A DataFrame containing performance metrics.
    """
    col_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    metrics = []

    for model, model_name in zip(models, model_names):
        metrics.append(get_result(model, X_train, y_train, X_test, y_test))

    return pd.DataFrame(metrics, columns=col_names, index=model_names)

# 2-3. Draw ROC curve
def draw_roc_curve(models, model_names, X_test, y_test, title="ROC Curve"):
    """
    Plot ROC curves for multiple models.

    Args:
        models (list): List of machine learning models.
        model_names (list): List of model names.
        X_test (array-like): Test data features.
        y_test (array-like): Test data labels.
        title (str): Title for the ROC curve plot.
    """
    plt.figure(figsize=(10, 10))

    for model, model_name in zip(models, model_names):
        if hasattr(model, "predict_proba"):
            pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            pred_proba = model.decision_function(X_test)
        else:
            raise ValueError("Model does not have a predict_proba or decision_function method.")

        fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
        plt.plot(fpr, tpr, label=model_name)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()

