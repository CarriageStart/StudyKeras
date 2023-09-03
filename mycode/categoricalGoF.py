import numpy as np
from scipy import stats

x = np.array([90,30,35,55,40])    #관찰도수
e_x = np.array([0.3,0.15,0.1,0.25,0.2])*250   #기대도수

def main() :
    global x, e_x
    bResult = testGoodnessOfFit(x, e_x)
    if (bResult) :
        print("Null hypothesis cannot be rejected.")
        return
    print("Null hypothesis is rejected.")


def testGoodnessOfFit(data, expect):
    '''
    '   One-way chi-square test
    '       * Null hypothesis : The categoprical data has the given frequencies.
    '
    '   Parameters :
    '       f_obs : array like
    '       f_exp : array like (=None)  * If None, mean of f_obs is used.
    '       ddof  : array like (=0)     * d.o.f of Chisquaer = len(f_obs) - 1 - ddof
    '
    '   Return :
    '       chisq : chisquare value
    '       pvalue: p-value of chisquare value
    '''
    ret = stats.chisquare(data, expect)
    print(ret)
    print("D.o.F : ", ret.df)
    if ret.pvalue < 0.05 :
        return False
    return True


if __name__ == "__main__" :
    main()
