import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


filepath = "../data/promotion.csv"

def main() -> None :
    profit_promoted, profit_not_promoted = retrieveData()
    leveneResult = leveneVarTest(profit_promoted, profit_not_promoted)


def retrieveData() -> (pd.DataFrame, pd.DataFrame) :
    global filepath
    data = pd.read_csv(filepath)

    print(data.head(5))
    print("Profit of the promoted")
    print(data[data["promotion"]=="YES"]["profit"].describe())
    print(data[data["promotion"]=="NO"]["profit"].describe())

    profit_promoted = data[data["promotion"]=="YES"]["profit"]
    profit_not_promoted  = data[data["promotion"]=="NO"]["profit"]
    return profit_promoted, profit_not_promoted


def leveneVarTest(data1: pd.DataFrame, data2: pd.DataFrame) \
        -> stats._morestats.LeveneResult :
    return stats.levene(data1, dfata2)
    


if __name__ == "__main__" :
    main()

