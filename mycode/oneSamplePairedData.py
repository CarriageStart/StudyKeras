import numpy as np
import pandas as pd
from scipy import stats

dataX = np.array([224,270,400,444,590,660,1400,680])
dataY = np.array([116,96,239,329,437,597,689,576])


def main():
    global dataX, dataY

    dict_data = {'before': dataX, 'after': dataY}
    data = pd.DataFrame(data=dict_data)
    print(data.describe())
    print(f"Data : \n{data}")

    bResult = testRelativeDifference(dataX, dataY)
    if bResult :
        print("Test is doesn't meaningful")
    else :
        print("Test rejects the hypothesis")


def testRelativeDifference(dataA, dataB):
    '''
    '   T-test on two "Related" scores of samples 
    '       * Null hypothesis : Two related samples have identical mean.
    '
    '   Parameters:
    '       a, b        : array_like
    '       alternative : {'two-sided', 'less', 'greater'} (='two-sided')
    '                   => a 'less' than b
    '                   => a 'greater' than b
    '
    '   Returns :
    '       statistic   : value of test statistic
    '       pvalue      : p-value of the test statistic
    '       df          : Degree of freedom of t-Statistic
    '''
    result = stats.ttest_rel(dataA, dataB)
    print(result)
    if (result.pvalue < 0.05):
        return False
    return True


if __name__ == "__main__" :
    main()
