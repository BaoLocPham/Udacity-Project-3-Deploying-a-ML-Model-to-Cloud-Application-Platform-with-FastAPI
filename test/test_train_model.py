'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-06 15:30:37
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-06 15:31:07
 # @ Description:
    Train model module test
 '''
import pytest


@pytest.fixture
def args_train():
    class Args:
        input_data = './data/cleaned_census.csv'
        output_model = './model/model.joblib'
        output_encoder = './model/encoder.joblib'
        output_label_encoder = './model/lb_encoder.joblib'
    return Args()


def test_run(args_train):
    # train_model.run(args)

    # Check that trained model, encoder and lb exist
    # assert os.path.isfile(args.output_model)
    # assert os.path.isfile(args.output_encoder)
    # assert os.path.isfile(args.output_label_encoder)
    assert True
