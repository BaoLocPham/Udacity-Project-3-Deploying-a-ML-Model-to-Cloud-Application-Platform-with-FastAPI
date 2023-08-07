'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-05 16:39:45
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-07 22:45:58
 # @ Description:
    Features and Categorical feature scripts
'''


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

FEATURES = [
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

LABEL = 'salary'


def get_features():
    return FEATURES


def get_cat_features():
    return CAT_FEATURES
