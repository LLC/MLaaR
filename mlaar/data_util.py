# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import pickle
from IPython.display import Image, display, HTML
from scipy.stats import spearmanr
import math


def get_corr_and_pvals(X, Y):
    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)
    corrs = []
    pvals = []
    for c in X.columns:
        notnull_index = X[c].notnull()
        if(X[c].dtype.name == 'category'):
            corr, pval = spearmanr(X[c][notnull_index].cat.codes, Y[notnull_index], nan_policy='omit')
        else:
            corr, pval = spearmanr(X[c][notnull_index], Y[notnull_index], nan_policy='omit')
        corrs.append(corr)
        pvals.append(pval)

    get_nondiag = np.vectorize(lambda x: x[0] != x[1])
    k = pd.DataFrame({"name": X.columns, "corrs": corrs, "pvals": pvals,
                      "corrs_abs": np.abs(corrs)})[['name', 'corrs', 'pvals', 'corrs_abs']]
    k = k.sort_values(by='pvals', ascending=True)
    #display(k[:20].style.format({'pvals': '{:.4f}', 'corrs': '{:.4f}', 'corrs_abs': '{:.2f}'}))
    return k


def to_categorical(df, columns):
    for c in columns:
        df[c] = df[c].astype("category")
    return df


def count(df):
    return len(df)


def count_nonNaN(df):
    T = pd.DataFrame(df.count()).reset_index()
    T.columns = ['name', 'nonNA_count']
    L = len(df)
    T['NA_count'] = L - T['nonNA_count']
    return T


# show data types
def get_types(df):
    T = pd.DataFrame(df.dtypes).reset_index()
    T.columns = ['name', 'type']
    return T



def get_basic_info_show(df):
    type_info = get_types(df)
    non_info = count_nonNaN(df)
    basic_info = df.describe(include='all').T.reset_index().rename(
        columns={"index": "name"})
    mid1 = pd.merge(type_info, non_info, on='name', how='left')
    mid2 = pd.merge(mid1, basic_info, on='name', how='left')
    memory_usage = np.sum(df.memory_usage())/1e3
    display(HTML("<h2>memory usage:</h2> <h3>%.03f kb</h3>" %
                 (np.sum(df.memory_usage())/1e3)))

    def show_category(row, max_len=10):
        k = row.value_counts(dropna=False).reset_index()
        # header        
        name = k.columns[1]

        # table
        k = k.sort_values(by=name, ascending=False)
        total = np.sum(k[name])
        k['ratio'] = k[name] / total

        if len(k) > 0:
            display(HTML("<h2>%s</h2>" % name))
            v = k.reset_index(drop=True).T
            display(v)
            return

    if len(df.dtypes == "category") >0:
        df[df.dtypes[df.dtypes == "category"].index].apply(show_category, axis=0)

    display(mid2.fillna(""))
    return mid2




def get_basic_info(df):
    type_info = get_types(df)
    non_info = count_nonNaN(df)
    basic_info = df.describe(include='all').T.reset_index().rename(
        columns={"index": "name"})
    mid1 = pd.merge(type_info, non_info, on='name', how='left')
    mid2 = pd.merge(mid1, basic_info, on='name', how='left')
    memory_usage = np.sum(df.memory_usage())/1e3

    def show_category(row, max_len=10):
        k = row.value_counts(dropna=False).reset_index()
        # header        
        name = k.columns[1]

        # table
        k = k.sort_values(by=name, ascending=False)
        total = np.sum(k[name])
        k['ratio'] = k[name] / total

        if len(k) > 0:
            display(HTML("<h2>%s</h2>" % name))
            v = k.reset_index(drop=True).T
            display(v)
            return

    return mid2, memory_usage


def check_numeric_info(df):
    return df.describe().T.reset_index()


def check_category_info(df):
    display(df.describe(include=['category']).T.reset_index())


# ref: https://pandas.pydata.org/pandas-docs/stable/categorical.html
def check_category(df, max_size, cols=None):
    def check_category_info(row):
        print(row.unique())
        display(pd.DataFrame(row.value_counts()[:max_size]))

    if cols is None:
        cols = df.columns[df.dtypes == "category"]

    df[cols].apply(check_category_info, axis=0)


if __name__ == "__main__":
    train_dataset = pd.read_csv(os.path.join("../data/titanic/train.csv"))
    print("資料總筆數 %d" % count(train_dataset))