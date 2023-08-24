import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno

import PySimpleGUI as sg


file_paths = [
        "../data/titanic/train.csv",
        "../data/titanic/test.csv",
        "../data/titanic/gender_submission.csv"
]


def preprocess() -> (pd.DataFrame, pd.DataFrame) :
    global file_paths
    train_ds = pd.read_csv(file_paths[0])
    test_ds = pd.read_csv(file_paths[1])
    gender_ds = pd.read_csv(file_paths[2])

    # Order the test_ds like train_ds
    test_ds["Survived"] = gender_ds["Survived"] # Copy into new Series
    test_ds = test_ds[train_ds.columns]

    print("train_ds Shape : ", train_ds.shape)
    print("train_ds Header : ", train_ds.columns)
    print(train_ds.head(6))
    print("\n")
    print("test_ds Shape : ", test_ds.shape)
    print("test_ds Header : ", test_ds.columns)
    print(test_ds.head(6))
    print("\n\n")
    
    # Show "NaN" graphically
    #msno.matrix(train_ds, figsize=(16, 8))
    #msno.matrix(test_ds, figsize=(16, 8))
    #plt.show()

    print("Before dropping Nan")
    print(train_ds.isnull().sum())
    print(test_ds.isnull().sum())

    train_ds = train_ds.drop(["Cabin"], axis=1).dropna()
    test_ds = test_ds.drop(["Cabin"], axis=1).dropna()

    print("After dropping Nan")
    print(train_ds.isnull().sum())
    print(test_ds.isnull().sum())
    return train_ds, test_ds


def main() -> None :
    train_ds, test_ds = preprocess()
    train_ds.info()
    train_stats = show_stats(train_ds)
    sg.Print(train_stats.to_string()print)


def show_stats(table: pd.DataFrame) -> pd.DataFrame :
    skew = []
    kurtosis = []
    null = []
    median = []

    for val in table.columns :
        if table[val].dtype.str == "|O" : continue
        median.append(table[val].median())
        skew.append(table[val].skew())
        null.append(table[val].isnull().sum())
        kurtosis.append(table[val].kurtosis())

    table_stats = table.describe().T
    table_stats["skew"] = skew
    table_stats["kurtosis"] = kurtosis
    table_stats["median"] = median
    table_stats["null"] = null

    return table_stats

if __name__ == "__main__" :
    main()

    





