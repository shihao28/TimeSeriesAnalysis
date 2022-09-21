import torch
from torch import nn, optim
from src.model_dl import *


class ConfigDL(object):

    # Training config
    train = dict(

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),

        data=dict(
            data_path="data/01_raw/RestaurantVisitors.csv",
            label="total",
            # Only list the features that are to be used
            numeric_features=[
                "rest1", "rest2", "rest3", "rest4",
            ],
            category_features=[
                "holiday",
            ],
            datetime_features=dict(
                # key is column name, value is datetime format
                date="%m/%d/%Y",
            ),
            freq="D"
        ),

        train_val_test_split=dict(
            split_ratio=0.3,
        ),

        model={
            # n_out in model =n_steps_out
            RNN.__name__: RNN(2, 64, 1, 1),
        },

        model_setting={
            "n_steps_in": 4,
            "n_steps_out": 1,
            "exog_var": ['holiday'],  # Set it to None if no exog_var
            # use_seasonal=True,  # whether to use sarima
            # seasonal_period=7,  # for sarimax
        },

        criterion=nn.MSELoss(),

        epochs=1,

        batch_size=32,

        optimizer=dict(
            alg=optim.SGD,
            param=dict(
                lr=0.01, momentum=0.9, weight_decay=0.0005
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
            registered_model_name="my_cls_model",
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
