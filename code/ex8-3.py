import pandas as pd
data=pd.read_csv('ex8-3.csv')
data.head()

import statsmodels.api as sm
import statsmodels.formula.api as smf
fit=smf.ols('y~program+C(number)+program*C(number)', data).fit()
sm.stats.anova_lm(fit)
