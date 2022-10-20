from scipy import stats
import numpy as np
import robust
import matplotlib.pyplot as plt


def outliers_zscore(data_all, i, thr=3, modified=False, show=False):
    """
    Definition for removing outliers using the Z-score method
    Threshold set by default for points of std > 3
    modified=True will set the modified zscore method, which should work better for non-gaussia
    """

    # Initial parameters
    data = data_all[i]          # One data series
    size = len(data)            # Size of data

    # Loop for the next iterations - stopped when no outlers
    while True:
        if not modified:
            data_z = stats.zscore(data)
        else:
            mad = robust.mad(data)
            median = np.median(data)
            data_z = .6745 * (data - median) / mad

        # original data
        plt.plot(data_z, 'o')
        plt.title(f"z-score outliers removal: data[{i}]")
        plt.xlabel('Data index')
        plt.ylabel('Z distance')

        # threshold
        plt.plot([0, size], [thr, thr], 'r--')
        plt.plot([0, size], [-thr, -thr], 'r--')

        # are there outliers?
        to_remove = []
        for i, el in enumerate(data_z):
            if el > thr or el < -thr:
                to_remove.append(i)

        # marking to_removed with red cross
        for el in to_remove:
            plt.plot(el, data_z[el], 'rx', markersize=12)

        plt.show() if show else plt.close()

        print(f"z-score outliers removal: data[{i}]", f": removed {len(to_remove)} elements")
        if len(to_remove) == 0:
            break
        else:
            data = np.delete(data, list(reversed(to_remove)))

    return data
