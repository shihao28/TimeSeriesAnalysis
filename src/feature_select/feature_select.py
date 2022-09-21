from sklearn.feature_selection import *


class FeatureSelect:
    def __init__(
        self, data, label, numeric_features_names,
        category_features_names):

        # Shift features
        mask = ~data.columns.isin([label])
        data.loc[:, mask] = data.loc[:, mask].shift(1)
        self.data = data.iloc[1:, ]

        self.label = label
        self.numeric_features_names = numeric_features_names
        self.category_features_names = category_features_names

        # Get numeric and category features
        self.numeric_features = self.data[self.numeric_features_names]
        self.category_features = self.data[self.category_features_names]

    def select_features(self):
        f_stat, f_stat_pvalues = f_regression(
            self.data[self.numeric_features_names + self.category_features_names], self.data[self.label])
        f_stat_pvalues = {column_name:pvalue for column_name, pvalue in zip(self.numeric_features_names + self.category_features_names, f_stat_pvalues)}
        f_stat_pvalues = {k: f_stat_pvalues[k] for k in sorted(f_stat_pvalues, key=f_stat_pvalues.get, reverse=False)}
 
        return f_stat_pvalues
