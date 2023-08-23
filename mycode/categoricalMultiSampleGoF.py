
import numpy as np
import pandas as pd
from scipy import stats

time=np.tile(['day', 'evening', 'night'],2)
count=np.array([905, 890, 870, 45, 55, 70])
goods=np.repeat(['O', 'X'],3)   #양품:O,  불량품:X


def main() :
    data={'time':time, 'goods':goods, 'count':count}
    d_table = pd.crosstab(
        index=data["goods"],
        columns=data["time"],
        values=data["count"],
        aggfunc=sum,
        margins=True, margins_name="total"
    )
    print("Data table : \n", d_table)

    p_table = pd.crosstab(
        index=data["goods"],
        columns=data["time"],
        values=data["count"],
        aggfunc=sum,
        margins=True, margins_name="total",
        normalize="index"
    ).round(4)
    print("Probability table : \n", p_table)

    chi, p, df, expected = stats.chi2_contingency(d_table)
    print(f" chi : {chi:.4f},\n p : {p:.4f},\n df : {df},\n expected :\n {expected}")
    if (p < 0.05) :
        print("The Null Hypothesis is rejected")
    print("The Null Hypothesis cannot be rejected")

if __name__ == "__main__" :
    main()

