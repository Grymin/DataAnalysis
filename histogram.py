import matplotlib.pyplot as plt
import numpy as np


class Histogram:

    def __init__(self, data, iq=0, bins_arbitrary=None, printed=True):
        # Imported
        self.data = data
        self.len = len(data)
        self.iq = iq                    #
        self.printed = printed          # Results will be printed

        # Declared
        self.max = np.max(data)         # Max in the data
        self.min = np.min(data)         # Min in the data

        # Bins
        if bins_arbitrary is None:
            self.number_of_bins()
        else:
            self.bins = bins_arbitrary
        if printed:
            print(f"Number of bins: k = {self.bins}")
        self.hist_drawing()

    def number_of_bins(self):
        """Calculates number of the bins
        It can be automated by command
        plt.hist(bins= 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt')
        Here it has been introduced for the purpose of comparison of the results"""

        # self.bin = math.ceil(self.wid / (10 ** self.mag))
        # self.dx = math.ceil((self.wid / (10 ** self.mag)) / self.bin) * 10 ** self.mag
        # self.dens = 1
        # self.beg = round(math.floor(self.min / self.dx) * self.dx, 4)
        # self.end = round(math.ceil(self.max / self.dx) * self.dx, 4)

        # Sturge's rule
        bins_sturge = int(np.ceil(np.log2(self.len)))+1
        # Freedman-Diaconis rule
        fda_h = 2*self.iq / (self.len ** (1/3))
        bins_fda = int(np.ceil((self.max - self.min) / fda_h))
        self.bins = bins_fda

        # print('      square-root: {}'.format(math.sqrt(leng)))
        # print('      rices formula: {}'.format(2 * leng ** (1 / 3)))
        # print('      3.3: {}'.format(math.ceil(3.3 * math.log10(len(self.data)) + 1)))

        if self.printed:
            print("*"*20)
            print(f"Size of data: {self.len}")
            print(f"number of bins:")
            print(f"\tSturge: {bins_sturge}")
            print(f"\tFreedman-Diaconis: {bins_fda}")

    def hist_drawing(self):
        y, x = np.histogram(self.data, self.nbins)
        x = (x[:-1] + x[1:]) / 2
        plt.plot(x, y)

        plt.plot([self.mean, self.mean], [0, max(y)], 'b--', label='mean')
        plt.plot([self.median, self.median], [0, max(y)], 'r--', label='median')
        plt.plot([self.mode, self.mode], [0, max(y)], 'g--', label='mode')
        plt.plot([self.mean - self.std, self.mean + self.std], [0.1 * max(y), 0.1 * max(y)], 'c', label='std')

        # Graph
        plt.legend()
        plt.show()