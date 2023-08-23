import pandas as pd
data=pd.read_csv('ex8-1.csv')
data.head()

data.groupby("aggregate").y.describe()

import statsmodels.api as sm
import statsmodels.formula.api as smf
fit=smf.ols('y~aggregate', data).fit()

sm.stats.anova_lm(fit, typ=1)
