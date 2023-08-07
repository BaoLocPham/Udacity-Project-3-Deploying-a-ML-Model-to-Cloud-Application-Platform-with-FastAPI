'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-06 16:30:13
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-06 16:30:56
 # @ Description:
    Pytest fixture configuration
 '''


import pytest
import joblib
import pandas as pd


@pytest.fixture
def raw_data():
    return pd.read_csv("./data/census.csv")


@pytest.fixture
def cleaned_data():
    return pd.read_csv("./data/test_cleaned_census.csv")


@pytest.fixture(scope='session', autouse=True)
def model():
    model_path = "./model/model.joblib"
    model = joblib.load(model_path)
    return model


@pytest.fixture(scope='session', autouse=True)
def encoder():
    encoder_path = "./model/encoder.joblib"
    encoder = joblib.load(encoder_path)
    return encoder


@pytest.fixture(scope='session', autouse=True)
def label_encoder():
    lb_encoder_path = "./model/lb_encoder.joblib"
    label_encoder = joblib.load(lb_encoder_path)
    return label_encoder


@pytest.fixture(scope='session', autouse=True)
def valid_cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture(scope='session', autouse=True)
def valid_features():
    return [
        "age",
        "hours-per-week",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture(scope='session', autouse=True)
def valid_columns():
    return [
        "age",
        "hours-per-week",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "salary"
    ]
