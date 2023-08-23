import pandas as pd
from scipy import stats

file_path = "../data/ex7-4.csv"

def main() :
    global file_path
    data=pd.read_csv(file_path)
    print(data.head())

    print("Data table")
    table_data = pd.crosstab(
        index=data["amount"],
        columns=data["level"],
        values=data['count'],
        aggfunc=sum,
        margins=True,
        margins_name="Total"
    )
    print(table_data)

    print("Probability table")
    table_prob = pd.crosstab(
        index=data["amount"],
        columns=data["level"],
        values=data["count"],
        aggfunc=sum,
        margins=True,
        margins_name="Total",
        normalize="index"
    ).round(4)
    print(table_prob)

    print("Start chisquare test")
    chi, p, df, expected = stats.chi2_contingency(table_data)
    print(expected)
    table_expected = pd.DataFrame(
        data=expected,
        index=table_data.index,
        columns=table_data.columns
    )
    print("Expectation table")
    print(table_expected)

    if p < 0.05 :
        print("Data is not independent.")
    else :
        print("Data is independent.")
    print(chi, p)



if __name__ == "__main__" :
    main()

