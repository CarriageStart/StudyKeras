import numpy as np
from scipy import stats


x = np.array([2000, 1975, 1900, 2000, 1950, 1850, 1950, 2100, 1975])

def main():
    '''
    '   T tess for the man of "One" sample
    '       * Null Hypothesis : The expected mean of a sample is equalt to 'popmean'
    '
    '   Parameters :
    '       a           : array_like
    '       popmean     : mean
    '       alternative : {'two-sided', 'less', 'greater'} (='two-sided')
    '
    '   Return :
    '       statistic   : Value of test statistic
    '       pvalue      : p-value of corresponding test statistic
    '       df          : Degree of freedom of t statistic.
    '''
    global x
    popmean = 1951
    result = stats.ttest_1samp(x, popmean=popmean)
    print(result)
    if (result.pvalue < 0.05) :
        print(f"The mean is not seemed to be {popmean}.")
    else :
        print(f"The mean is similar to {popmean}.")


if __name__ == "__main__" :
    main()
