import numpy as np
import pandas as pd
from scipy import stats


data = [
    23, 23, 24, 27, 29, 31, 32, 33, 33, 35,
    36, 37, 40, 42, 43, 43, 44 ,45, 48, 48,
    54, 54, 56, 57, 57, 58, 58, 58, 58, 59,
    61, 61, 62, 63, 64, 65, 66, 68, 68, 70,
    73, 73, 74, 75, 77, 81, 87, 89, 93, 97
]

def main() :
    global data, numBins
    bins, counts = calculateBins(data)

    mean = np.mean(data)
    std = np.std(data)
    # Chisquare test
    counts_data = np.concatenate([[0], counts, [0]], dtype="float64")
    counts_gaus = fillGausian(bins, mean, std)
    ratio = np.sum(counts_data) / np.sum(counts_gaus)
    counts_gaus = counts_gaus * ratio
    result_chi = stats.chisquare(counts_data, counts_gaus)
    print("Start chisquare test")
    print(f"Bins : {bins}")
    print(f"Observation : {counts_data}")
    print(f"Expectaion : {counts_gaus}")
    print(result_chi)
    if (result_chi.pvalue < 0.05) :
        print(f"The distribution doesn't follow Norm({mean:.1f}, {std:.2f})")
    else :
        print(f"The distribution seems to follow Norm({mean:.1f}, {std:.2f})")

    # Kolmogorov_Smirnov(K-S) test
    '''
    '   Kolmogorov_Smirnov test for goodness of fit
    '       * Null hypothesis : c.d.f. of the sample is identical to the c.d.f. of expectation.
    '           => Can be one sample and one dist, or two samples
    '           => To make c.d.f., the category value of the sample should be continuous.
    '           => Test statistics : D = max_x(|F(x) - G(x)|)
    '             * If N.H is right, the difference should follow normal if F and G are properly normed.
    '
    '   Parameters :
    '       rvs : array or str, callablle
    '       cdf : array or str, callablle
    '       args: tuple, sequence (optional)
    '           => arguments for
    '
    '   Return :
    '       well..
    '''
    print("Start K-S test")
    result = stats.kstest(data, "norm", args=(mean, std))
    print(result)
    if result.pvalue < 0.05 :
        print(f"The distribution doesn't follow Norm({mean:.1f}, {std:.2f})")
        return
    print(f"The distribution seems to follow Norm({mean:.1f}, {std:.2f})")


def fillGausian(bins, mean, std, n=1000000) :
    size_bins = bins.size
    var_gaus = stats.norm.rvs(mean, std, n)
    counts = np.zeros(shape=size_bins+1, dtype="int32")
    for val in var_gaus:
        if val < bins[0] :
            counts[0] += 1
        elif val > bins[size_bins-1] :
            counts[size_bins] += 1
        index = findBins(val, bins) + 1
        counts[index] += 1
    return counts.astype(dtype="float64") / n

    
def findBins(val, bins) :
    i = 0
    while i < bins.size :
        if val < bins[i] :
            break
        i += 1
    return i - 1


def calculateBins(data):
    max_numBins = len(data)
    min_data, max_data = min(data), max(data)+1
    min_binWidth = 1

    min_count = 0
    numBins = max_numBins
    bins = np.linspace(min_data, max_data, num=numBins)
    counts = np.zeros(shape=bins.size-1, dtype="int32")
    while True :
        for i in range(bins.size - 1) :
            for x in data :
                if x < bins[i] :
                    continue
                elif x > bins[i+1] :
                    break
                counts[i] += 1
        min_count = np.min(counts)
        if (min_count >= 5) : 
            break

        numBins -= 1
        bins = np.linspace(min_data, max_data, num=numBins)
        counts = np.zeros(shape=bins.size-1)
    return bins, counts


if __name__ == "__main__" :
    main()
