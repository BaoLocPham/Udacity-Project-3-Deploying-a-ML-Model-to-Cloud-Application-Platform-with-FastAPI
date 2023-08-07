'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-06 21:41:12
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-06 21:42:05
 # @ Description:
    Model evaluate on slice data
'''


import argparse
import logging
import pandas as pd
import joblib
from starter.ml.data import process_data
from starter.features import get_cat_features
from sklearn.model_selection import train_test_split
from starter.ml.model import compute_model_metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def run(
    args=None,
    input_data: str = None,
    output_model: str = None,
    output_encoder: str = None,
    output_label_encoder: str = None,
    output_slice: str = None
):
    """The model evaluation function"""

    if args is not None:
        (input_data, output_model, output_encoder,
         output_label_encoder, output_slice) = (args.input_data,
                                                args.output_model,
                                                args.output_encoder,
                                                args.output_label_encoder,
                                                args.output_slice)

    df = pd.read_csv(input_data)
    _, test = train_test_split(df, test_size=0.20)

    model = joblib.load(output_model)

    encoder = joblib.load(output_encoder)

    lb = joblib.load(output_label_encoder)
    logger.warning(test.columns)
    # define slice values
    slice_values = []
    for cat in get_cat_features():
        for _class in test[cat].unique():
            df_temp = test[test[cat] == _class]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=get_cat_features(),
                label="salary",
                encoder=encoder,
                lb=lb,
                training=False
            )

            y_preds = model.predict(X_test)

            precision_score, recall_score, f1_score = compute_model_metrics(
                y_test, y_preds)

            line = f"[{cat}->{_class}] Precision: {precision_score} " \
                   f"Recall: {recall_score} F1: {f1_score}"
            logging.info(line)
            slice_values.append(line)

    # compute overall metrics
    logger.info("Computing overall metrics...")
    _X_test, _y_test, _, _ = process_data(
        test,
        categorical_features=get_cat_features(),
        label="salary", encoder=encoder, lb=lb, training=False
    )
    _y_preds = model.predict(_X_test)
    _precision_score, _recall_score, _f1_score = compute_model_metrics(
        _y_test, _y_preds)
    line = f"[Overall Score] - Precision: {_precision_score} " \
        f"Recall: {_recall_score} F1: {_f1_score}"
    logging.info(line)
    slice_values.append(line)

    with open(output_slice, 'w') as file:
        for slice_value in slice_values:
            file.write(slice_value + '\n')


if __name__ == "__main__":
    # Create an argument parser with description
    parser = argparse.ArgumentParser(description="This steps train model")

    parser.add_argument("--input_data", type=str,
                        default="./data/cleaned_census.csv",
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

    parser.add_argument("--output_slice", type=str,
                        # required=True,
                        default="./model/slice.txt",
                        help="Path to save the sclice evaluate file")
    args = parser.parse_args()

    # Retrieve the arguments and perform further processing
    input_data_file = args.input_data
    output_model_file = args.output_model
    output_encoder_file = args.output_encoder
    output_label_encoder_file = args.output_label_encoder
    output_slice_file = args.output_slice
    # Parse the arguments using the given parser
    args = parser.parse_args()

    # Call the go function with the parsed arguments
    _ = run(args)
