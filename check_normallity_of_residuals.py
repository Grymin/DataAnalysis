import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro


def check_norm_of_res(data):
    """Definition to check if residuals in the data matrix are normally distributed."""

    # Initial parameters
    data_res = []       # List of residuals in all the series

    # TODO: importing df and calculating residuals as seperate column to account for?
    # Calculating residuals
    for i in range(len(data)):
        avg = sum(data[i]) / len(data[i])
        res = data[i] - avg
        data_res.extend(res)

    # Shapiro-Wilk test for the residuals
    # TODO: not too good on large samples. Jarque-Bera or Omnibus to implement for such a case?
    stat, pval = shapiro(data_res)
    normal_res = True if pval > 0.05 else False
    print(f"RESIDUALS: stat: {stat}, pval: {pval}, normal: {normal_res}")

    # Shapiro-Wilk test for each data series
    for i in range(len(data)):
        stat, pval = shapiro(data[i])
        normal = True if pval > 0.05 else False
        print(f"DATA[{i}]: stat: {stat}, pval: {pval}, normal: {normal}")

    # Histogram: RESIDUALS
    fig, axes = plt.subplots(2,2)
    fig.tight_layout(pad=5)
    sns.histplot(data=pd.DataFrame(data_res, columns=["Residuals"]), x='Residuals', kde=True,
                 ax=axes[0,0]).set(title=f"Histogram of the residuals")

    # TODO graph of residuals of entire data or residuals of one series?
    # https://towardsdatascience.com/statistical-hypothesis-testing-with-python-6a2f38c12486
    """    sns.kdeplot(data=df, x=dv, hue=iv,
                fill=False, ax=ax)
    plt.show()
    pg.normality(df, dv=dv, group=iv, method='shapiro')"""

    # Histogram: data series
    for i in range(len(data)):
        graph_title = f"Histogram of the residuals"
        sns.kdeplot(data=pd.DataFrame(data[i], columns=['Values']), x='Values', ax=axes[0, 1]).set(title=graph_title)

    # Q-Q plot - any std
    sm.qqplot(np.asarray(data_res), line='r', ax=axes[1, 0])

    # Q-Q plot - 45 deg (std=1)
    sm.qqplot(np.asarray(data_res), line='45', ax=axes[1, 1])
    plt.show()

    return normal




