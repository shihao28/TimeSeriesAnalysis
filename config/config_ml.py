import numpy as np


class ConfigML(object):

    # Training config
    train = dict(

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

            # insert missing date and ffill it
            ffill_missing_date=True
        ),

        train_val_test_split=dict(
            split_ratio=0.1,
        ),

        model=dict(
            start_p=0, start_q=0,
            max_p=6, max_q=6,
            # Set it to empty list if no exog_var
            exog_var=[
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
                'CBOT.ZC_Settle_nearby_shift31', 'CBOT.ZC_Settle_nearby_shift32',
                'CBOT.ZC_Settle_nearby_shift33', 'CBOT.ZC_Settle_nearby_shift34',
                'CBOT.ZC_Settle_nearby_shift35', 'CBOT.ZC_Settle_nearby_shift36',
                'CBOT.ZC_Settle_nearby_shift37', 'CBOT.ZC_Settle_nearby_shift38',
                'CBOT.ZC_Settle_nearby_shift39', 'CBOT.ZC_Settle_nearby_shift40',
                'CBOT.ZC_Settle_nearby_shift41', 'CBOT.ZC_Settle_nearby_shift42',
                'CBOT.ZC_Settle_nearby_shift43', 'CBOT.ZC_Settle_nearby_shift44',
                'CBOT.ZC_Settle_nearby_shift45', 'CBOT.ZC_Settle_nearby_shift46',
                'CBOT.ZC_Settle_nearby_shift47', 'CBOT.ZC_Settle_nearby_shift48',
                'CBOT.ZC_Settle_nearby_shift49', 'CBOT.ZC_Settle_nearby_shift50',
                'CBOT.ZC_Settle_nearby_shift51', 'CBOT.ZC_Settle_nearby_shift52',
                'CBOT.ZC_Settle_nearby_shift53', 'CBOT.ZC_Settle_nearby_shift54',
                'CBOT.ZC_Settle_nearby_shift55', 'CBOT.ZC_Settle_nearby_shift56',
                'CBOT.ZC_Settle_nearby_shift57', 'CBOT.ZC_Settle_nearby_shift58',
                'CBOT.ZC_Settle_nearby_shift59', 'CBOT.ZC_Settle_nearby_shift60',
                ],
            use_seasonal=False,  # whether to use sarima
            seasonal_period=14,  # for sarimax
            # max_exo_var=2,  # -1 means use all
        ),

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
            tsa="r2",
        ),

        mlflow=dict(
            tracking_uri="http://127.0.0.1",
            backend_uri="sqlite:///mlflow.db",
            artifact_uri="./mlruns/",
            experiment_name="Best Time Series Pipeline",
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
