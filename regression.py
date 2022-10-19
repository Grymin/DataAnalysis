import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def simple_regression(iv_vector, dv_vector, print_residuals=True):
    """
    Caclulates simple regression for the data using simple regression in scipy
    :param iv_vector: vector of idependent variable
    :param dv_vector: vector of dependent variable
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


example_iv = np.arange(5, 9, 0.5)
example_dv = [40, 43, 45, 44, 48, 46, 51, 50]

simple_regression(example_iv, example_dv)