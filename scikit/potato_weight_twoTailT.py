import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

#%matplotlib inline

potato_weights = [
        89.03, 95.05, 88.26, 90.07, 90.6, 
        87.0, 87.67, 88.8, 90.46, 81.33
]

def main() -> None :
    result = two_sided_ttest()
    plot_result(result)
    

def two_sided_ttest(pop_mean=90) -> stats._stats_py.TtestResult :
    global potato_weights
    data = np.array(potato_weights)
    result = stats.ttest_1samp(data, pop_mean)
    print(type(result))
    print(result)
    print("result.statistic(t_value): %.3f, result.pvalue: %.4f" % (result.statistic, result.pvalue))
    print("result.df(d.o.f): %d" % result.df) # 10 - 1 : sample mean is used 
    print("result.count(?): %d" % result.df)
    print("result.index(?): %d" % result.df)
    interval = result.confidence_interval()
    print("Interval : (%.3f, %.3f)" % (interval.low, interval.high))
    return result


async def plot_result(result: stats._stats_py.TtestResult) -> None :
    global potato_weights
    ax = sns.distplot(
            potato_weights, kde=False, fit=stats.norm, label="potato_chip", color="blue", rug=True
    )
    """
    with displot or histplot, you need to fit and draw normal distribution yourself.
    ex)
    mu, std = stats.norm.fit(data)      # Note : actually, mu and std are sample mean and std... 
                                        #       mu == data.mean(),  std == data.std()
    x = np.linspace(xmin, xmax, 100)
    fitted_norm = stats.norm.pdf(x, mu, std) 
    plt.hist(data, bins=25, density=True, color="g")
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()


    just use distplot...
    """
    ax.set(xlabel="Weights of one potato chip")
    plt.legend()
    plt.show()


if __name__ == "__main__" :
    main()


