from sklearn.feature_selection import *


class FeatureSelect:
    def __init__(
        self, data, label, numeric_features_names,
        category_features_names, shifted_numeric_features_names,
        shifted_category_features_names):

        self.data = data
        self.label = label
        self.numeric_features_names = numeric_features_names
        self.category_features_names = category_features_names
        self.shifted_numeric_features_names = shifted_numeric_features_names
        self.shifted_category_features_names = shifted_category_features_names

        # Get numeric and category features
        self.numeric_features = data[self.numeric_features_names]
        self.category_features = data[self.category_features_names]
        self.shifted_numeric_features = self.data[self.shifted_numeric_features_names]
        self.shifted_category_features = self.data[self.shifted_category_features_names]

    def select_features(self):
        f_stat, f_stat_pvalues = f_regression(
            self.data[
                self.numeric_features_names + self.category_features_names +\
                    self.shifted_numeric_features_names + self.shifted_category_features_names],
            self.data[self.label])
        f_stat_pvalues = {column_name:pvalue for column_name, pvalue in zip(self.numeric_features_names + self.category_features_names + self.shifted_numeric_features_names + self.shifted_category_features_names, f_stat_pvalues)}
        f_stat_pvalues = {k: f_stat_pvalues[k] for k in sorted(f_stat_pvalues, key=f_stat_pvalues.get, reverse=False)}

        return f_stat_pvalues
