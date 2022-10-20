import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


def simple_regression(iv_vector, dv_vector, print_residuals=True):
    print(type(iv_vector))
    """
    Caclulates simple regression for the data using simple regression in scipy
    :param iv_vector: vector of idependent variable, vector or 1D np array
    :param dv_vector: vector of dependent variable, vector or 1D np array
    :param print_residuals: boolean, true if residuals should be printed on graph
    :return: None
    """
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(iv_vector, dv_vector)
    print("\n", "*"*20)
    print(f"\tSlope: {slope}\n",
          f"\tintercept:{intercept}\n"
          f"\tCorrelation coefficient rvalue:{rvalue}\n"
          f"\tTwo-sided p-value pvalue: {pvalue},"
          f"ok! below 0.05" if pvalue <= 0.05 else "too high! over 0.05"
          f"\tstderr:{stderr}\n")

    # Calculating model values
    y_hat = np.vstack((np.ones(len(iv_vector)), iv_vector)).T @ [intercept, slope]

    plt.plot(iv_vector, dv_vector, 'ko')
    plt.plot(iv_vector, y_hat, 'r-')
    if print_residuals:
        for i in range(len(iv_vector)):
            plt.plot([iv_vector[i], iv_vector[i]], [dv_vector[i], y_hat[i]], 'b--')

    plt.show()


# Simple regression example
"""
example_iv = np.arange(5, 9, 0.5)
example_dv = [40, 43, 45, 44, 48, 46, 51, 50]
simple_regression(example_iv, example_dv)
"""


def multiple_regression(iv1_vector, iv2_vector, dv_matrix, iv2_limits):
    """
    Defininig multiple regression, with visualization for one of the variable with 3 groups
    :param iv1_vector:
    :param iv2_vector:
    :param dv_matrix:
    :param iv2_limits:
    :return:
    """

    # Boolean vectors: in a group or not?
    plotidx = iv2_vector <= iv2_limits[0]
    plotidx2 = np.logical_and(iv2_vector > iv2_limits[0], iv2_vector <= iv2_limits[1])
    plotidx3 = iv2_vector > iv2_limits[1]

    # Plotting the groups
    plt.plot(iv1_vector[plotidx], dv_matrix[plotidx], 'ro')
    plt.plot(iv1_vector[plotidx2], dv_matrix[plotidx2], 'ko')
    plt.plot(iv1_vector[plotidx3], dv_matrix[plotidx3], 'bo')
    plt.show()

    matrix = np.vstack((np.ones((len(iv1_vector))), iv1_vector, iv2_vector, iv1_vector*iv2_vector)).T

    multireg = sm.OLS(endog=dv_matrix, exog=matrix).fit()
    print(multireg.summary())


# Multiple regression example
"""
example_iv1 = np.linspace(11, 30, 20)
example_iv2 = np.tile(np.linspace(1, 5, 5), 4)
example_dv = []
for i in range(4):
    example_dv = np.hstack((example_dv, 10*np.ones(5)+np.linspace(1, 5, 5)*(i+1)+np.random.randn(5)))
print(example_dv)
print(example_iv1)
print(example_iv2)

group_limits_iv2 = [2, 3]
multiple_regression(example_iv1, example_iv2, example_dv, group_limits_iv2)
"""