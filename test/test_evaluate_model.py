'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-06 15:30:37
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-06 15:31:07
 # @ Description:
    Train model module test
 '''
import pytest
import os
from starter import evaluate_model


@pytest.fixture
def args_evaluate():
    class Args:
        input_data = './data/cleaned_census.csv'
        output_model = './model/model.joblib'
        output_encoder = './model/encoder.joblib'
        output_label_encoder = './model/lb_encoder.joblib'
        output_slice = "./model/slice.txt"
    return Args()


def test_run(args_evaluate):
    evaluate_model.run(args_evaluate)

    assert os.path.isfile(args_evaluate.output_slice)
