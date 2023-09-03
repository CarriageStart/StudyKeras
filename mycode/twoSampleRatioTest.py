
import pandas as pd
from scipy import stats

def main() :
    '''
    '   Fisehr exact test on 2x2 contingency table
    '       * Null hypothesis : Compare the true odds ratio of the populations
    '
    '   Parameters :
    '       table       : array_like of ints with (2, 2) shape
    '       alternative : {'two-sided', 'less', 'greater'} (='two-sided')
    '               => The first axis is 'less' than the second axis.
    '               => The first axis is 'greater' than the second axis.
    '
    '   Return :
    '       statistic   :
    '       pvalue      :
    '''
    data = pd.DataFrame(
            [[56, 24], [44, 37]],
            index=['A', 'B'],
            columns=['use', 'unuse']
    )
    print(data.describe())
    print(data)

    result = stats.fisher_exact(data, alternative='greater')
    print(result)
    if (result.pvalue < 0.05) :
        print("The null hypthesis is rejected")
    else :
        print("The null is not rejected")


if __name__ == "__main__" :
    main()

