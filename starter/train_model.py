'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-05 16:07:39
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-06 10:53:47
 # @ Description:
    Script to train machine learning model.
'''
import pandas as pd
import joblib
import logging
import argparse
from starter.features import get_cat_features
from starter.ml.data import process_data, simple_cleaning_data
from starter.ml.model import train_model
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


def run(
    args=None,
    input_data: str = None,
    output_model: str = None,
    output_encoder: str = None,
    output_label_encoder: str = None,
):
    if args is not None:
        input_data, output_model, output_encoder, output_label_encoder = \
            (args.input_data,
             args.output_model,
             args.output_encoder,
             args.output_label_encoder)
    data = pd.read_csv(input_data)

    # logger.info(data.columns)
    cleaned_data = simple_cleaning_data(data, label="salary")
    logger.warning(f"{cleaned_data['salary'].nunique()}")

    train, test = train_test_split(cleaned_data, test_size=0.20)

    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=get_cat_features(),
        label="salary", training=True
    )

    # Train and save a model.
    model = train_model(
        X_train=X_train,
        y_train=y_train
    )

    joblib.dump(model, f'{output_model}')
    joblib.dump(encoder, f"{output_encoder}")
    joblib.dump(lb, f"{output_label_encoder}")

    logger.info(f"Save model to {output_model}")
    logger.info(f"Save encoder to {output_encoder}")
    logger.info(f"Save label encoder to {output_label_encoder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument Parser for Input Data, Output Model, \
             Output Encoder, and Label Encoder")
    parser.add_argument("--input_data", type=str,
                        default="./data/census.csv",
                        # required=True,
                        help="Path to the input data file")
    parser.add_argument("--output_model", type=str,
                        # required=True,
                        default="./model/model.joblib",
                        help="Path to save the output model file")
    parser.add_argument("--output_encoder", type=str,
                        # required=True,
                        default="./model/encoder.joblib",
                        help="Path to save the output encoder file")
    parser.add_argument("--output_label_encoder", type=str,
                        # required=True,
                        default="./model/lb_encoder.joblib",
                        help="Path to save the label encoder file")

    args = parser.parse_args()

    # Retrieve the arguments and perform further processing
    input_data_file = args.input_data
    output_model_file = args.output_model
    output_encoder_file = args.output_encoder
    output_label_encoder_file = args.output_label_encoder
    run(
        args=None,
        input_data=input_data_file,
        output_model=output_model_file,
        output_encoder=output_encoder_file,
        output_label_encoder=output_label_encoder_file
    )
