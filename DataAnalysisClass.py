import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statistics
import robust
from _definicje import progressbar
from histogram import Histogram


class DataAnalysis:

    def __init__(self, data, printed=True):
        self.data = data
        self.printed = printed

        # Statistics
        self.min_val = None
        self.max_val = None
        self.mean = None
        self.median = None
        self.mode = None
        self.size = None
        self.std = None
        self.cv = None
        self.ci = None
        self.quartiles = None
        self.quartile1_numpy, self.quartile3_numpy = None, None
        self.iq = None
        self.central_moments = None

        self.statistics()

    def statistics(self):
        """Calculates and prints basic statistics"""
        self.min_val = np.min(self.data)
        self.max_val = np.max(self.data)
        self.mean = np.mean(self.data)
        self.median = np.median(self.data)
        self.mode = stats.mode(self.data)[0][0]
        self.size = len(self.data)
        self.std = np.std(self.data, ddof=1)        # ddof = 0 - biased std
        self.cv = self.std / self.mean
        self.quartiles = statistics.quantiles(self.data, n=4)
        self.quartile1_numpy = np.percentile(self.data, 25)
        self.quartile3_numpy = np.percentile(self.data, 75)
        self.iq = self.quartile3_numpy - self.quartile1_numpy

        print('*'*20)
        print("Main statistics:")
        print(f"\tMin: {self.min_val}")
        print(f"\tMax: {self.max_val}")
        print(f"\tMean: {self.mean}")
        if self.median == 0:
            print("\tToo high accuracy of result to determine median!")
        print(f"\tMedian: {self.median}")
        print(f"\tMode: {self.mode}")
        print(f"\tSize: {self.size}")
        print(f"\tStandard deviation Std: {self.std}")
        print(f"\tCoeff. of variation CV: {self.cv}")
        print(f"Falco, J.G. (2016). Applied statistics. Low dispersion: <15%, average: <30%, high: >=30%")
        print(f"mean is suitable for representing the data" if self.cv < 0.15 else "average dispersion" if self.cv < 0.3
              else "data has little statistical meaning")
        print(f"\tQuartiles are: {self.quartiles}")
        print(f"\tNumpy quartiles: {self.quartile1_numpy, self.quartile3_numpy}")
        print('*'*20)

    def confidence_interval_formula(self, confidence=95):
        """Calculations of confidence interval by definition"""
        citmp = (1-confidence / 100) / 2
        confint = self.mean + stats.t.ppf([citmp, 1-citmp], self.size) * self.std / np.sqrt(self.size)
        print("\n", "*" * 20)
        print("Confidence interval with formula: ", confint)
        print("*" * 20)

    def confidence_interval_bootstrapping(self, boots=100, confidence=95):
        """Calculations of confidence interval by bootstrapping
        Boots - number of repetitions of resampling with replacemt"""
        means = np.zeros(boots)
        for i in range(boots):
            means[i] = np.mean(np.random.choice(self.data, self.size))
        confint = [np.percentile(means, (100-confidence)/2), np.percentile(means, 100-(100-confidence)/2)]
        print("\n", "*" * 20)
        print("Confidence interval with bootstrapping: ", confint)
        print("*" * 20)

    def central_moment_discrete(self, r: int, progress=True, printed=True):
        """
        Calculates central moments
        r - order to calculate
        progress - if progress should be printed
        """

        self.central_moments = np.zeros(r + 1)

        for rzad in range(r + 1):
            value = 0.
            if progress:
                progressbar(rzad, r, opis=f'calculating moment for rzad {rzad} out of {r}')
            for i in range(self.size):
                value += (self.data[i] - self.mean) ** rzad
            self.central_moments[rzad] = value / (self.size - 1)  # TODO czy w momencie centralnym jest minus 1?

        if printed:
            print("\n\n", "*"*20)
            print("Central moments:")
            for rzad in range(r + 1):
                print(f"order {rzad}: {self.central_moments[rzad]}")
            print("*"*20)

        return self.central_moments

    def hist_outsorcing(self, nbins):
        Histogram(self.data, self.iq, bins_arbitrary=nbins)

    def data_density(self):
        """prints data density"""
        plt.plot(self.data, '.', color=[.5, .5, .5], label='Data points')

        plt.plot([1, self.size], [self.mean, self.mean], 'b--', label='mean')
        plt.plot([1, self.size], [self.median, self.median], 'r--', label='median')
        plt.plot([1, self.size], [self.mode, self.mode], 'g--', label='mode')

        # Graph
        plt.legend()
        plt.show()

    def qq_plot(self):
        """QQ plot to compare to normal distribution"""

        # normalization of the data and generation of the normal distribution
        data_normalized = np.sort(stats.zscore(self.data))      # zscore: (x - x_av)/std
        normal_dist = stats.norm.ppf(np.linspace(0, 1, self.size))

        # plotting
        plt.plot(data_normalized, normal_dist, 'o')

        # setting same axes
        xl, xr = plt.xlim()
        yl, yr = plt.ylim()
        minval, maxval = np.min([xl, xr, yl, yr]), np.max([xl, xr, yl, yr])
        lims = [minval, maxval]
        plt.xlim(lims)
        plt.ylim(lims)

        # plotting
        plt.plot(lims, lims)
        plt.xlabel("Theoretical normal distribution")
        plt.ylabel("Observed data")
        plt.title("QQ plot")
        plt.show()

    def violinplot(self):

        # # Seaborn - preferred for pands
        # seaborn.set(style='whitegrid')
        # ax = seaborn.violinplot(data=self.data)
        # plt.show()

        # Matplotlib
        plt.violinplot(self.data, showmeans=True, showextrema=True, showmedians=True)
        plt.title("Violin plot")
        plt.show()

    def outliers_zscore(self, thr=3, modified=False):
        """
        Definition for removing outliers using the Z-score method
        Threshold set by default for points of std > 3
        Modified = True will set the modified zscore method
        """

        # print
        if self.printed:
            print("\n", "*" * 20)
            print("OUTLIERS_ZSCORE")

        while True:
            if not modified:
                data_z = stats.zscore(self.data)
            else:
                mad = robust.mad(self.data)
                data_z = .6745 * (self.data-self.median) / mad

            # original data
            plt.plot(data_z, 'o')
            plt.title("z-score outliers removal")
            plt.xlabel('Data index')
            plt.ylabel('Z dist.')

            # threshold
            plt.plot([0, self.size], [thr, thr], 'r--')
            plt.plot([0, self.size], [-thr, -thr], 'r--')

            # are there outliers?
            to_remove = []
            for i, el in enumerate(data_z):
                if el > thr or el < -thr:
                    to_remove.append(i)

            # marking to_removed
            for el in to_remove:
                plt.plot(el, data_z[el], 'rx', markersize=12)

            plt.show()

            if len(to_remove) == 0:
                if self.printed:
                    print(f"no outliers for threshold of {thr}")
                    print("*"*20)
                break
            else:
                self.data = np.delete(self.data, list(reversed(to_remove)))
                print(f"removed {len(to_remove)} from the data, repeating the loop")

            # actualize statistics:
            self.statistics()

    def one_sample_t_test(self, val):
        t, p = stats.ttest_1samp(self.data, val)

        # # printing data
        plt.plot(self.data, 'o')
        plt.xlabel('Data index')
        plt.ylabel('Data value')
        plt.show()

        # H0 parameter distribution and observed t
        x = np.linspace(-4, 4, 101)
        tdist = stats.t.pdf(x, self.size-1) * np.mean(np.diff(x))
        plt.plot(x, tdist)
        plt.plot([t, t], [0, max(tdist)], 'r--')
        plt.legend('H0 distribution, observed t')
        plt.xlabel('t-value')
        plt.ylabel('pdf(t)')
        plt.title(f"t({self.size-1}) = {t}, p = {p}")
        plt.show()

        print("\n", "*" * 20)
        print(f"ONE_SAMPLE_T_TEST FOR VAL OF {val}")
        print(f"t: {t}, p: {p}")
        print("*" * 20)
