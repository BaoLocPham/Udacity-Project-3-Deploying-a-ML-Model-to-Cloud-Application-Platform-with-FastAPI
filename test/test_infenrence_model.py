'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-06 15:30:37
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-06 15:31:07
 # @ Description:
    Inference model module test
'''
import numpy as np
from starter import inference_model


def test_run(
    cleaned_data,
    model,
    encoder,
    label_encoder
):
    cleaned_data.drop("salary", axis=1, inplace=True)
    y_preds, y_preds_label = inference_model.run(
        cleaned_data,
        model,
        encoder,
        label_encoder
    )

    assert isinstance(y_preds, np.int64)
    assert isinstance(y_preds_label, str)
