import pandas as pd
import numpy as np
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import StandardScaler
from scipy import stats
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


class MissingValAnalysis:
    def __init__(
        self, data):
        self.data = data

    def __hypothesis_testing(self):
        # Generate missing value indicator
        missing_features_indicator = self.missing_indicator.fit_transform(self.data)
        missing_features_names = self.data.columns[self.missing_indicator.features_]
        missing_features_indicator = pd.DataFrame(
            missing_features_indicator, columns=missing_features_names)

        # Check type of missingness
        feature_a = []
        feature_b = []
        p_value = []
        missing_type = []
        for missing_feature_name, missing_feature in missing_features_indicator.iteritems():
            for feature_name, feature in self.data.iteritems():
                if missing_feature_name == feature_name:
                    continue

                # Binning continuous features
                if feature_name in self.numeric_features_names:
                    feature = pd.qcut(x=feature, q=10, labels=False)

                # Chi2 test
                ct = pd.crosstab(
                    missing_feature, feature, margins=False)
                _, p_val, _, _ = stats.chi2_contingency(ct)
                if p_val < 0.05:
                    missing_type.append("MAR")
                else:
                    missing_type.append("MCAR/ MNAR")
                feature_a.append(missing_feature_name)
                feature_b.append(feature_name)
                p_value.append(p_val)
        missingness_report = pd.DataFrame({
            "Feature_A": feature_a, "Feature_B": feature_b,
            "p-value": p_value, "Missing_Type": missing_type
        })
        final_missingness_report = pd.crosstab(missingness_report["Feature_A"], missingness_report["Missing_Type"])
        total_count = final_missingness_report.iloc[0].sum()
        final_missingness_report = final_missingness_report / total_count * 100
        final_missingness_report = final_missingness_report.idxmax(1)

        return missingness_report, final_missingness_report

    def generate_report(self):
        plt.ioff()

        # Replace empty string with nan
        self.data.replace("", np.nan, inplace=True)

        if self.scale:
            self.data[self.numeric_features_names] =\
                StandardScaler().fit_transform(
                    self.data[self.numeric_features_names])

        missing_columns = self.data.columns[self.data.isna().any(0)]
        missingness = msno.matrix(self.data)
        missingness_corr = msno.heatmap(self.data, cmap='rainbow')
        nonna_count_by_column = msno.bar(self.data)

        # Check MCAR
        # https://impyute.readthedocs.io/en/master/user_guide/diagnostics.html#little-s-mcar-test-1
        # msno.matrix(self.data.sort_values('age'))

        # Check MAR
        first_lvl_index = np.array([["missing"]*8, ["complete"]*8]).reshape(-1)
        second_lvl_index = np.tile(self.data.describe().index.values, 2)
        index = [first_lvl_index, second_lvl_index]
        mar = dict()
        for missing_column in missing_columns:
            missing_portion = self.data[self.data[missing_column].isna()].drop(missing_column, axis=1)
            complete_portion = self.data[~self.data[missing_column].isna()]
            mar[missing_column] = pd.concat([missing_portion.describe(), complete_portion.describe()], 0)
            mar[missing_column].index = index

        # Check MNAR

        # Conduct hypothesis testing to check type of missingness
        missingness_report, final_missingness_report = self.__hypothesis_testing()

        plt.ion()

        return missingness_report, final_missingness_report

    def impute_missing_val(self, missing_column):
        # https://www.machinelearningplus.com/time-series/time-series-analysis-python/

        # Forward Fill
        self.data = self.data.ffill()

        # Backward Fill
        # self.data = self.data.bfill()

        # Linear Interpolation
        # self.data['row_num'] = np.arange(len(self.data))
        # data_nona = self.data.dropna(subset=[missing_column])
        # f = interp1d(data_nona['row_num'], data_nona[missing_column])
        # self.data[missing_column] = f(self.data['row_num'])

        # Cubic Interpolation
        self.data['row_num'] = np.arange(len(self.data))
        data_nona = self.data.dropna(subset=[missing_column])
        f = interp1d(data_nona['rownum'], data_nona['value'], kind='cubic')
        self.data[missing_column] = f(self.data['row_num'])

        # Interpolation References:
        # https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
        # https://docs.scipy.org/doc/scipy/reference/interpolate.html

        # kNN mean
        def knn_mean(ts, n):
            out = np.copy(ts)
            for i, val in enumerate(ts):
                if np.isnan(val):
                    n_by_2 = np.ceil(n/2)
                    lower = np.max([0, int(i-n_by_2)])
                    upper = np.min([len(ts)+1, int(i+n_by_2)])
                    ts_near = np.concatenate([ts[lower:i], ts[i:upper]])
                    out[i] = np.nanmean(ts_near)
            return out

        self.data[missing_column] = knn_mean(self.data[missing_column].values, 8)

        # Seasonal Mean
        def seasonal_mean(ts, n, lr=0.7):
            """
            Compute the mean of corresponding seasonal periods
            ts: 1D array-like of the time series
            n: Seasonal window length of the time series
            """
            out = np.copy(ts)
            for i, val in enumerate(ts):
                if np.isnan(val):
                    ts_seas = ts[i-1::-n]  # previous seasons only
                    if np.isnan(np.nanmean(ts_seas)):
                        ts_seas = np.concatenate([ts[i-1::-n], ts[i::n]])  # previous and forward
                    out[i] = np.nanmean(ts_seas) * lr
            return out

        self.data[missing_column] = seasonal_mean(self.data[missing_column], n=12, lr=1.25)
