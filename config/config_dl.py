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

            # Only list the features that are to be used for eda
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

            # insert missing date and ffill it
            ffill_missing_date=True
        ),

        train_val_test_split=dict(
            split_ratio=0.1,
        ),

        model={
            # n_out in model =n_steps_out
            RNN.__name__: RNN(
                in_dim=32, hidden_dim=64, num_layers=1, n_out=10),
        },

        model_setting=dict(
            # whether to scale data
            scale=True,
            n_steps_in=30,  # 30, 60, 90
            n_steps_out=10,
            # Set it to empty list if no exog_var
            exog_var=dict(
                numeric=[
                    # 'Week_no_sin', 'Week_no_cos',
                    'Month_sin', 'Month_cos',
                    'CBOT.ZC_Settle_nearby_shift1', 'CBOT.ZC_Settle_nearby_shift2',
                    'CBOT.ZC_Settle_nearby_shift3', 'CBOT.ZC_Settle_nearby_shift4',
                    'CBOT.ZC_Settle_nearby_shift5', 'CBOT.ZC_Settle_nearby_shift6',
                    'CBOT.ZC_Settle_nearby_shift7', 'CBOT.ZC_Settle_nearby_shift8',
                    'CBOT.ZC_Settle_nearby_shift9', 'CBOT.ZC_Settle_nearby_shift10',
                    'CBOT.ZC_Settle_nearby_shift11', 'CBOT.ZC_Settle_nearby_shift12',
                    'CBOT.ZC_Settle_nearby_shift13', 'CBOT.ZC_Settle_nearby_shift14',
                    'CBOT.ZC_Settle_nearby_shift15', 'CBOT.ZC_Settle_nearby_shift16',
                    'CBOT.ZC_Settle_nearby_shift17', 'CBOT.ZC_Settle_nearby_shift18',
                    'CBOT.ZC_Settle_nearby_shift19', 'CBOT.ZC_Settle_nearby_shift20',
                    'CBOT.ZC_Settle_nearby_shift21', 'CBOT.ZC_Settle_nearby_shift22',
                    'CBOT.ZC_Settle_nearby_shift23', 'CBOT.ZC_Settle_nearby_shift24',
                    'CBOT.ZC_Settle_nearby_shift25', 'CBOT.ZC_Settle_nearby_shift26',
                    'CBOT.ZC_Settle_nearby_shift27', 'CBOT.ZC_Settle_nearby_shift28',
                    'CBOT.ZC_Settle_nearby_shift29', 'CBOT.ZC_Settle_nearby_shift30',
                    # 'CBOT.ZC_Settle_nearby_shift31', 'CBOT.ZC_Settle_nearby_shift32',
                    # 'CBOT.ZC_Settle_nearby_shift33', 'CBOT.ZC_Settle_nearby_shift34',
                    # 'CBOT.ZC_Settle_nearby_shift35', 'CBOT.ZC_Settle_nearby_shift36',
                    # 'CBOT.ZC_Settle_nearby_shift37', 'CBOT.ZC_Settle_nearby_shift38',
                    # 'CBOT.ZC_Settle_nearby_shift39', 'CBOT.ZC_Settle_nearby_shift40',
                    # 'CBOT.ZC_Settle_nearby_shift41', 'CBOT.ZC_Settle_nearby_shift42',
                    # 'CBOT.ZC_Settle_nearby_shift43', 'CBOT.ZC_Settle_nearby_shift44',
                    # 'CBOT.ZC_Settle_nearby_shift45', 'CBOT.ZC_Settle_nearby_shift46',
                    # 'CBOT.ZC_Settle_nearby_shift47', 'CBOT.ZC_Settle_nearby_shift48',
                    # 'CBOT.ZC_Settle_nearby_shift49', 'CBOT.ZC_Settle_nearby_shift50',
                    # 'CBOT.ZC_Settle_nearby_shift51', 'CBOT.ZC_Settle_nearby_shift52',
                    # 'CBOT.ZC_Settle_nearby_shift53', 'CBOT.ZC_Settle_nearby_shift54',
                    # 'CBOT.ZC_Settle_nearby_shift55', 'CBOT.ZC_Settle_nearby_shift56',
                    # 'CBOT.ZC_Settle_nearby_shift57', 'CBOT.ZC_Settle_nearby_shift58',
                    # 'CBOT.ZC_Settle_nearby_shift59', 'CBOT.ZC_Settle_nearby_shift60',
                   
                ],
                category=[
                    # 'Month'
                    # 'Plantation', 'Pollination', 'Harvest'
                ]
            ),
        ),

        criterion=nn.MSELoss(),

        epochs=50,

        batch_size=64,

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
            experiment_name="Best Time Series Pipeline",
            run_name="trial",
            registered_model_name="my_tsa_model_dl",
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
