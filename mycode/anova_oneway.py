import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def main() :
    data = pd.read_csv("../data/ex8-1.csv")
    print(data.head(5))

    print("data.groupby('aggregate') : \n", data.groupby("aggregate"))
    print(data)
    print("data.groupby('aggregate').y : \n", data.groupby("aggregate").y)
    print("data.groupby('aggregate').y.describe() : \n", data.groupby('aggregate').y.describe())

    fit = smf.ols("y ~ aggregate", data).fit()
    print(type(fit))
    print(fit)

    result = sm.stats.anova_lm(fit, typ=1)
    print(type(result))
    print(result)


if __name__ == "__main__" :
    main()



