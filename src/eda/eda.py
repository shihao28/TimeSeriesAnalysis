import logging
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot
import numpy as np


class EDA:
    def __init__(
        self, data, label, numeric_features_names,
        category_features_names, datetime_features_names,
        scale=True):

        self.data = data
        self.label = label
        self.numeric_features_names = numeric_features_names
        self.category_features_names = category_features_names
        self.datetime_features_names = datetime_features_names
        self.scale = scale
        self.num_plot_per_fig = 4
        self.scale_alg = None

        # Get numeric and category features
        self.numeric_features = self.data[self.numeric_features_names]
        self.category_features = self.data[self.category_features_names]
        self.datetime_features = self.data[self.datetime_features_names]

    def __general_analysis(self):
        numeric_features_count = len(self.numeric_features_names)
        category_features_count = len(self.category_features_names)
        self.data.replace("", np.nan, inplace=True)
        missing_value_percent = (self.data.isna().sum() / len(self.data) * 100).round(2)

        general_analysis = pd.DataFrame({
            "dtype": self.data.dtypes, "MissingV_Value_Percent": missing_value_percent
        })
        general_analysis["Features_Type"] = ""
        general_analysis.loc[general_analysis["Features_Type"].index.isin(self.numeric_features_names), "Feature_Type"] = "Numeric"
        general_analysis.loc[general_analysis["Features_Type"].index.isin(self.category_features_names), "Feature_Type"] = "Category"

        general_analysis_fig, general_analysis_ax = plt.subplots()
        general_analysis_ax.axis('tight')
        general_analysis_ax.axis('off')
        general_analysis_ax.table(
            cellText=general_analysis.values,
            colLabels=general_analysis.columns,
            rowLabels=general_analysis.index, loc='center', fontsize=200)

        return general_analysis_fig

    def __univariate_analysis(self, numeric_features_dropna, category_features_dropna):
        # numeric features
        numeric_features_stats = numeric_features_dropna.agg([
            "min", "max", "mean", "median", "std",
            "skew", "kurtosis"])
        numeric_features_quantile = numeric_features_dropna.quantile([0.25, 0.75])
        numeric_features_stats = pd.concat(
            [numeric_features_stats, numeric_features_quantile], 0)
        numeric_features_stats.sort_values(
            by="std", axis=1, ascending=True, inplace=True)
        numeric_features_stats = numeric_features_stats.round(4)
        numeric_features_stats_fig, numeric_features_stats_ax = plt.subplots()
        numeric_features_stats_ax.axis('tight')
        numeric_features_stats_ax.axis('off')
        numeric_features_stats_ax.table(
            cellText=numeric_features_stats.values,
            colLabels=numeric_features_stats.columns,
            rowLabels=numeric_features_stats.index, loc='center', fontsize=20)

        # Kde plot
        kdeplot_all = []
        for i, (name, column) in enumerate(numeric_features_dropna.iteritems()):
            if i % self.num_plot_per_fig == 0:
                kdeplot_fig, kdeplot_ax = plt.subplots(
                    int(self.num_plot_per_fig**0.5),
                    int(self.num_plot_per_fig**0.5))
            row_idx, col_idx = divmod(
                i % self.num_plot_per_fig, int(self.num_plot_per_fig**0.5))
            column.plot.kde(
                ax=kdeplot_ax[row_idx, col_idx], secondary_y=True, title=name)
            # ax_tmp.text(0.5, 0.5, "test", fontsize=22)
            if (i + 1) % self.num_plot_per_fig == 0 or i == len(numeric_features_dropna.columns)-1:
                kdeplot_all.append(kdeplot_fig)
                plt.tight_layout()

        # Boxplot
        boxplot_all = []
        fig, ax = plt.subplots()
        for i, (name, column) in enumerate(numeric_features_dropna.iteritems()):
            if i % self.num_plot_per_fig == 0:
                boxplot_fig, boxplot_ax = plt.subplots(
                    int(self.num_plot_per_fig**0.5),
                    int(self.num_plot_per_fig**0.5))
            row_idx, col_idx = divmod(
                i % self.num_plot_per_fig, int(self.num_plot_per_fig**0.5))
            boxplot_ax[row_idx, col_idx].boxplot(column, 0, 'gD')
            boxplot_ax[row_idx, col_idx].set_title(f"{name}")
            if (i + 1) % self.num_plot_per_fig == 0 or i == len(numeric_features_dropna.columns)-1:
                boxplot_all.append(boxplot_fig)
                plt.tight_layout()

        # Category features
        # Freq plot
        freqplot_all = []
        for i, (name, column) in enumerate(category_features_dropna.iteritems()):
            if i % self.num_plot_per_fig == 0:
                freqplot_fig, freqplot_ax = plt.subplots(
                    int(self.num_plot_per_fig**0.5),
                    int(self.num_plot_per_fig**0.5))
            row_idx, col_idx = divmod(
                i % self.num_plot_per_fig, int(self.num_plot_per_fig**0.5))
            bar = freqplot_ax[row_idx, col_idx].bar(
                column.value_counts().index, column.value_counts().values)
            freqplot_ax[row_idx, col_idx].bar_label(bar)
            freqplot_ax[row_idx, col_idx].set_title(f"{name}")
            if (i + 1) % self.num_plot_per_fig == 0 or i == len(category_features_dropna.columns)-1:
                freqplot_all.append(freqplot_fig)
                plt.tight_layout()

        return numeric_features_stats_fig, kdeplot_all, boxplot_all, freqplot_all

    def __bivariate_analysis(self, numeric_features_dropna, category_features_dropna):
        # Categorical vs Categorical
        cat_vs_cat_plot_all = []
        if self.problem_type == "classification":
            columns_comb = list(combinations(category_features_dropna.columns, 2))
            for i, (column_a, column_b) in enumerate(columns_comb):
                category_features_crosstab = pd.crosstab(
                    category_features_dropna[column_a],
                    category_features_dropna[column_b],
                    margins=True, values=category_features_dropna[self.label],
                    aggfunc=pd.Series.count, normalize="all")
                category_features_crosstab.fillna(0, inplace=True)
                if i % self.num_plot_per_fig == 0:
                    cat_vs_cat_plot_fig, cat_vs_cat_plot_ax = plt.subplots(
                        int(self.num_plot_per_fig**0.5),
                        int(self.num_plot_per_fig**0.5))
                row_idx, col_idx = divmod(
                    i % self.num_plot_per_fig, int(self.num_plot_per_fig**0.5))
                sns.heatmap(
                    category_features_crosstab,
                    ax=cat_vs_cat_plot_ax[row_idx, col_idx],
                    annot=True, cbar=True, fmt=".2%")
                cat_vs_cat_plot_ax[row_idx, col_idx].set(
                    title=f"{column_a} vs {column_b} with {self.label} as count")
                if (i + 1) % self.num_plot_per_fig == 0 or i == len(columns_comb)-1:
                    cat_vs_cat_plot_all.append(cat_vs_cat_plot_fig)
                    plt.tight_layout()

        # Numerical vs Categorical
        num_vs_cat_plot_all = []
        if self.problem_type == "classification":
            numeric_features_and_label = pd.concat(
                [numeric_features_dropna, category_features_dropna[self.label]], 1)
            for i, (name, column) in enumerate(numeric_features_dropna.iteritems()):
                if i % self.num_plot_per_fig == 0:
                    num_vs_cat_plot_fig, num_vs_cat_plot_ax = plt.subplots(
                        int(self.num_plot_per_fig**0.5),
                        int(self.num_plot_per_fig**0.5))
                row_idx, col_idx = divmod(
                    i % self.num_plot_per_fig, int(self.num_plot_per_fig**0.5))
                sns.boxplot(
                    x=self.label, y=column, data=numeric_features_and_label,
                    ax=num_vs_cat_plot_ax[row_idx, col_idx])
                if (i + 1) % self.num_plot_per_fig == 0 or i == len(columns_comb)-1:
                    num_vs_cat_plot_all.append(num_vs_cat_plot_fig)
                    plt.tight_layout()

        # Numerical vs Numerical
        # correlation plot
        corr_matrix = numeric_features_dropna.corr(method="pearson")
        corr_fig, corr_ax = plt.subplots()
        sns.heatmap(corr_matrix, ax=corr_ax, annot=True)
        corr_ax.set(title="Correlation Matrix (Pearson)")

        num_vs_num_plot_all = []
        columns_comb = list(combinations(numeric_features_dropna.columns, 2))
        for i, (column_a, column_b) in enumerate(columns_comb):
            if i % self.num_plot_per_fig == 0:
                num_vs_num_plot_fig, num_vs_num_plot_ax = plt.subplots(
                    int(self.num_plot_per_fig**0.5),
                    int(self.num_plot_per_fig**0.5))
            row_idx, col_idx = divmod(
                i % self.num_plot_per_fig, int(self.num_plot_per_fig**0.5))
            num_vs_num_plot_ax[row_idx, col_idx].scatter(
                x=numeric_features_dropna[column_a],
                y=numeric_features_dropna[column_b],
                c=category_features_dropna[self.label if self.problem_type == "classification" else None])
            num_vs_num_plot_ax[row_idx, col_idx].set(
                xlabel=column_a, ylabel=column_b,
                title=f"Plot of {column_b} vs {column_a}")
            if (i + 1) % self.num_plot_per_fig == 0 or i == len(columns_comb)-1:
                num_vs_num_plot_all.append(num_vs_num_plot_fig)
                plt.tight_layout()

        return cat_vs_cat_plot_all, num_vs_cat_plot_all, corr_fig, num_vs_num_plot_all

    def __save_to_pdf(self, *args):

        pp = PdfPages('eda.pdf')
        for arg in args:
            if isinstance(arg, matplotlib.figure.Figure):
                pp.savefig(arg)
            else:
                for fig in arg:
                    pp.savefig(fig)
        pp.close()

    def __time_series_analysis(self):
        # Overall plot
        overall_fig, overall_ax = plt.subplots()
        overall_ax.plot(self.datetime_features.iloc[:, 0], self.data[self.label], )
        overall_ax.set(
            xlabel='Datetime', ylabel=self.label,
            title=f"Plot of {self.label}")

        # Plot by year
        yearwise_fig, yearwise_ax = plt.subplots(1, 2)
        sns.lineplot(
            data=self.data, x='Month', y=self.label, hue='Year',
            ax=yearwise_ax[0])
        sns.boxplot(
            data=self.data, x='Month', y=self.label, hue='Year',
            ax=yearwise_ax[1])

        # Additive Decomposition
        result_add = seasonal_decompose(
            self.data[self.label], model='additive', extrapolate_trend='freq',
            period=12)
        # Multiplicative Decomposition
        result_mul = seasonal_decompose(
            self.data[self.label], model='multiplicative',
            extrapolate_trend='freq', period=12)
        # Plot
        result_add.plot().suptitle('Additive Decompose', fontsize=22)
        result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)

        # acf and pacf plot
        acf_50 = acf(self.data[self.label], nlags=50)
        pacf_50 = pacf(self.data[self.label], nlags=50)
        cf_fig, cf_ax = plt.subplots(1, 2)
        plot_acf(self.data[self.label], lags=50, ax=cf_ax[0])
        plot_pacf(self.data[self.label], lags=50, ax=cf_ax[1])

        # lag plot
        lag_fig, lag_ax = plt.subplots(1, 4)
        for i, ax in enumerate(lag_ax.flatten()[:4]):
            lag_plot(self.data[self.label], lag=i+1, ax=ax)
            ax.set_title('Lag ' + str(i+1))
        lag_fig.suptitle(f'Lag Plot of {self.label}')

    def __stationarity_test(self):
        adf_stat, pvalue, _, _, _, _ = adfuller(self.data[self.label])
        logging.info(f'p-value is {pvalue:.4f}')
        if pvalue < 0.05:
            logging.info('The series is stationary')
        else:
            logging.info('The series is not stationary')

        return None

    def __get_sample_entropy(self, U, m, r):
        # https://en.wikipedia.org/wiki/Sample_entropy
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

        def _phi(m):
            x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
            C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
            return sum(C)

        N = len(U)
        forecastability =  -np.log(_phi(m+1) / _phi(m))
        logging.info(f'Forecastability is {forecastability}')

        return forecastability

    def __granger_causality_test(self, maxlag=12):
        """
        The Null hypothesis is: the series in the second column, does not Granger cause the series in the first.
        """
        granger_causality_pvalue = dict()
        for column_name in self.numeric_features_names:
            granger_causality_pvalue[column_name] = grangercausalitytests(
                self.data[[self.label, column_name]], maxlag=maxlag)

        return granger_causality_pvalue

    def generate_report(self):
        self.__time_series_analysis()
        p_value = self.__stationarity_test()
        forecastability = self.__get_sample_entropy(
            self.data[self.label], m=2, r=0.2*np.std(self.data[self.label]))
        granger_causality_pvalue = self.__granger_causality_test(maxlag=12)

        return None