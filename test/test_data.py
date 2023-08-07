'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-06 20:44:23
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-06 21:07:13
 # @ Description:
    Data module test
'''
from starter.ml import data


def test_simple_cleaning_data(raw_data, valid_columns):
    cleaned_data = data.simple_cleaning_data(raw_data)
    assert set(cleaned_data.columns.to_list()) == set(valid_columns)


def test_process_data(
        cleaned_data,
        valid_cat_features,
        valid_features
):

    X_train, y_train, encoder, lb = data.process_data(
        cleaned_data,
        categorical_features=valid_cat_features,
        label="salary",
        training=True
    )

    assert len(X_train) > 0
    assert len(y_train) == len(X_train)
    assert encoder is not None
    assert lb is not None
