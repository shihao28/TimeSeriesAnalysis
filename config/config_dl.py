import numpy as np
import torch
from torch import nn, optim

from src.model_dl import *


class ConfigDL(object):

    # Training config
    train = dict(

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),

        data=dict(
            data_path="data/03_primary/nearby_prices.csv",
            label="CBOT.ZS_Settle_nearby",

            # Only list the features that are to be used
            numeric_features=[
                'CBOT.ZC_Settle_nearby',
                'Month_sin', 'Month_cos',
                'Day_sin', 'Day_cos',
                'Day_of_week_sin', 'Day_of_week_cos',
                'Week_no_sin', 'Week_no_cos',
            ],
            category_features=[
                # 'Year',
                'Month',
                'Day',
                'Day_of_week',
                'Week_no',
            ],
            datetime_features=dict(
                # key is column name, value is datetime format
                # strptime documentation
                # https://docs.python.org/3/library/datetime.html
                date="%Y-%m-%d",
            ),
            freq="D",
            shift_numeric_features={
                'CBOT.ZC_Settle_nearby': np.arange(60) + 1,
            },
            shift_category_features={

            },
        ),

        train_val_test_split=dict(
            split_ratio=0.1,
        ),

        model={
            # n_out in model =n_steps_out
            RNN.__name__: RNN(4, 64, 1, 1),
        },

        model_setting={
            "n_steps_in": 40,
            "n_steps_out": 1,
            # Set it to empty list if no exog_var
            "exog_var": dict(
                numeric=[
                    'Month_sin', 'Month_cos',
                    'CBOT.ZC_Settle_nearby_shift1'
                ],
                category=[
                    'Month'
                ]
            ),
        },

        criterion=nn.MSELoss(),

        epochs=1,

        batch_size=32,

        optimizer=dict(
            alg=optim.SGD,
            param=dict(
                lr=0.001, momentum=0.9, weight_decay=0.0005
            )
        ),

        lr_scheduler=dict(
            alg=optim.lr_scheduler.StepLR,
            param=dict(
                step_size=20, gamma=0.1
            )
        ),

        # WIP
        # tuning={

        # },

        evaluation=dict(
            # Get list of metrics from below
            # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            tsa="r2",
        ),

        mlflow=dict(
            tracking_uri="http://127.0.0.1",
            backend_uri="sqlite:///mlflow.db",
            artifact_uri="./mlruns/",
            experiment_name="Best Pipeline",
            run_name="trial",
            registered_model_name="my_tsa_model",
            port="5000",
        ),

        seed=0

    )

    # Prediction config
    predict = dict(

        data_path="data/01_raw/housing.csv",

        mlflow=dict(
            tracking_uri="http://127.0.0.1",
            backend_uri="sqlite:///mlflow.db",
            artifact_uri="./mlruns/",
            model_name="my_cls_model",
            port="5000",
            model_version="latest"
        ),
    )
