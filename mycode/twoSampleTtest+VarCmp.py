
import numpy as np
import pandas as pd
from scipy import stats

# Data
data = [
    [2.0,2.0,2.3,2.1,2.4],
    [2.2,1.9,2.5,2.3,2.4]
]

byPass = True 

def main():
    global data, byPass
    x = np.concatenate(data)
    mark = np.repeat(np.array(['A', 'B']), 5)
    d = {'mark': mark, 'data': x}
    data = pd.DataFrame(data=d)
    print(data)
    data.groupby("mark").data.describe()

    dataA = data[data.mark=='A']
    dataB = data[data.mark=='B']
    print(dataA)
    dataA.describe()
    print(dataB)
    dataB.describe()

    # Check whether std can be shared bet 'A' and 'B'.
    bResult = testNormality(x) # Normality Test
    if byPass :
        print("Levene test starts.")
        bResult = testLevene(dataA, dataB)
        print("Bartlett test starts.")
        bResult = testBartlett(dataA, dataB)
        print("Start the Pooled t-test for mean-equality.")
        bResult = testWithPooledStd(dataA, dataB)
        print("Start the diff-var t-test for mean-equality.")
        bResult = testWithTwoStd(dataA, dataB)
        return 

    if not bResult :
        print("The data doesn't show Normality!\nLevene test starts.")
        bResult = testLevene(dataA, dataB)
    else :
        print("The data show Normality!\nBartlett test starts.")
        bResult = testBartlett(dataA, dataB)

    if (bResult) :
        print("The data can be considered to have the same Var")
        print("Start the Pooled t-test for mean-equality.")
        bResult = testWithPooledStd(dataA, dataB)
        if bResult :
            print("Statistically, null hypothesis is not rejected")
        else :
            print("Statistically, null hypothesis is rejected")
    else : 
        print("The data cannot be considered to have the same Var")
        print("Start the independent t-test for mean-equality.")
        bResult = testWithTwoStd(dataA, dataB)
        if bResult :
            print("Statistically, null hypothesis is not rejected")
        else :
            print("Statistically, null hypothesis is rejected")



def testWithTwoStd(dataA, dataB):       # T test for means of two samples.
    '''
    '   T test for the mean of two independent samples.
    '       * Null hypothesis : Two independent samples hav identical mean.
    '   Parameters
    '       a, b        : Array likes with the same shape
    '       axis        : axis of the data
    '       equal_val   : pooled or Welch's t-test  (=True)
    '       nan_policy  : How to handle NaNs. {'propagate', 'omit', 'raise'}
    '       alternative : {'two-sided', 'less', 'greater'} (='two-sided')
    '       trim        : Yuen's T test
    '   Return
    '       statistic   : Value of test statistics(t value)
    '       pvalue      : pvalue
    '       df          : Degree of freedom of t Statistics
    '''
    result = stats.ttest_ind(
            dataA.data,
            dataB.data,
            alternative='two-sided',
            equal_var=False
    )
    print(result)
    if (result.pvalue < 0.05) :
        return False
    return True


def testWithPooledStd(dataA, dataB):        # T test for means of two samples with similar variance.
    '''
    '   T test for the mean of two independent samples.
    '       * Null hypothesis : Two independent samples hav identical mean.
    '   Parameters
    '       a, b        : Array likes with the same shape
    '       axis        : axis of the data
    '       equal_val   : pooled or Welch's t-test  (=True)
    '       nan_policy  : How to handle NaNs. {'propagate', 'omit', 'raise'}
    '       alternative : {'two-sided', 'less', 'greater'} (='two-sided')
    '       trim        : Yuen's T test
    '   Return
    '       statistic   : Value of test statistics(t value)
    '       pvalue      : pvalue
    '       df          : Degree of freedom of t Statistics
    '''
    result = stats.ttest_ind(dataA.data, dataB.data, equal_var=True)
    print(result)
    if (result.pvalue < 0.05) :
        return False
    return True


def testNormality(data):    # Normality test
    '''
    '   Shapiro-Wilk Test of normality. 
    '       * Null hypothesis : data follows the normal distribution.
    '       * Test statistics : ratio between ordered distribution and normal distribution.
    '   
    '   input : Array like
    '   return : scipy.stats._morestas.ShapiroResult
    '   return value can be use as 
    '       res = shapiro(data)
    '       value_testStatistic = res.statistic
    '       value_pValue = res.pvalue
    '       print(res)      # This prints the test statistic and p value.
    '''
    normality = stats.shapiro(data)
    print(normality)
    if (normality.pvalue < 0.05):
        return False
    return True


def testBartlett(dataA, dataB):     # F test for Variance of two samples with normality.
    '''
    '   Bartlett Test of normality. 
    '       * Null hypothesis : all input samplses are from (may-be) different population
    '                           with equal variances. 
    '               => It assumes the samples are retrieved from normal populations.
    '   
    '   inputs : 
    '       sample1, sample2, ... : 1D Array likes
    '
    '   return : scipy.stats._morestas.BartlettResult
    '   return value can be use as 
    '       res = bartlett(data)
    '       value_testStatistic = res.statistic
    '       value_pValue = res.pvalue
    '       print(res)      # This prints the test statistic and p value.
    '''
    equality = stats.bartlett(dataA.data, dataB.data)
    print(equality)
    if (equality.pvalue < 0.05) :
        return False
    return True

def testLevene(dataA, dataB):   # F test for variance of two samples without normality
    '''
    '   Levene Test of normality. 
    '       * Null hypothesis : all input samplses are from (may-be) different population
    '                           with equal variances. 
    '               => It assumes the samples are retrieved from normal populations.
    '   
    '   inputs : 
    '       sample1, sample2, ... : 1D Array likes
    '       center : {'mean', 'median', 'trimmed'} (='median')
    '       proportiontocut : float (=0.05)
    '
    '   return : scipy.stats._morestas.LeveneResult
    '   return value can be use as 
    '       res = bartlett(data)
    '       value_testStatistic = res.statistic
    '       value_pValue = res.pvalue
    '       print(res)      # This prints the test statistic and p value.
    '''
    equality = stats.levene(dataA.data, dataB.data)
    print(equality)
    if (equality.pvalue < 0.05) :
        return False
    return True


if __name__=="__main__":
    main()
