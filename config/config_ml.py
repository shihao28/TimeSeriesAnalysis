import pandas as pd


class ConfigML(object):

    # Training config
    train = dict(

        data=dict(
            data_path="data/01_raw/RestaurantVisitors.csv",
            label="total",
            numeric_features=[
                "rest1", "rest2", "rest3", "rest4",
            ],
            category_features=[
                "weekday", "holiday", "holiday_name"
            ],
            datetime_features=dict(
                # key is column name, value is [datetime format
                date="%m/%d/%Y",
                )
        ),

        train_val_test_split=dict(
            split_ratio=0.3,
        ),

        # model={
        #     SVC.__name__: SVC(probability=True),
        #     LogisticRegression.__name__: LogisticRegression()
        # },

        # tuning={
        #     "tune": False,
        #     "search_method": GridSearchCV,  # RandomizedSearchCV, BayesSearchCV
        #     SVC.__name__: dict(
        #         model__C=[1, 5],
        #         model__kernel=["linear", "poly", "rbf"]
        #         ),
        #     LogisticRegression.__name__: dict(
        #         model__penalty=["none", "l2"]
        #     )
        # },

        evaluation=dict(
            # Get list of metrics from below
            # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            regression="r2",
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
