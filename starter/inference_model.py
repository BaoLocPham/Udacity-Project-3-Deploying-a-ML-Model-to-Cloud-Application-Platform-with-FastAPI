'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-06 10:52:55
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-06 10:54:07
 # @ Description:
 '''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from starter.ml.data import process_data
from starter.ml.model import inference
from starter.features import get_cat_features


def run(
        data: pd.DataFrame,
        model: RandomForestClassifier = None,
        encoder=None,
        lb=None):
    """Inference data pipeline. Preprocess data and return predicts result

    Inputs
    ------
    data : pd.DataFrame
    model : sklearn.ensemble.RandomForestClassifier
        Model RandomForestClassifier
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.

    Returns
    -------
    y_preds: np.array
        Predicted values
    y_preds_label: np.array
        Predicted values in text

    """
    # process input user data
    X, _, _, _ = process_data(
        data,
        categorical_features=get_cat_features(),
        encoder=encoder, lb=lb, training=False)
    # print(X)
    y_preds = inference(model, X)
    # print(y_preds)
    y_preds_label = lb.inverse_transform(y_preds)
    return y_preds[0], y_preds_label[0]
