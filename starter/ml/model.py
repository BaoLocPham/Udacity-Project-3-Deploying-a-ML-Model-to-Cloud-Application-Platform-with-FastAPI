'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-05 16:07:39
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-06 10:01:01
 # @ Description:
    Machine learning model handler scripts
 '''

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
import logging

# Optional: implement hyperparameter tuning.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    scores = cross_val_score(model,
                             X_train,
                             y_train,
                             scoring='accuracy',
                             cv=cv, n_jobs=-1)
    logger.info('Accuracy on CV: %.4f (%.4f)' %
                (np.mean(scores), np.std(scores)))
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model
    using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    if not model:
        raise ValueError("Model RandomForestClassifier is none")
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("Model is not type of RandomForestClassifier")
    y_preds = model.predict(X)
    return y_preds
