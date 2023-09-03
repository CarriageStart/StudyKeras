import pandas as pd
data=pd.read_csv('ex8-2.csv')
data.head()   #A : 기계   B: 작업자   Y: 작업 속도

import statsmodels.api as sm
import statsmodels.formula.api as smf
fit=smf.ols('Y~C(A)+C(B)', data).fit()
sm.stats.anova_lm(fit, typ=1)

