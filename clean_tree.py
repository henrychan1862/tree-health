"""Module for cleaning 2015StreetTreesCensus_TREES.csv
Make sure you have unzipped and pasted the .csv file in the same folder"""
import numpy as np
import pandas as pd


def read_tree():
    # import 2015StreetTreesCensus_TREES.csv
    trees = pd.read_csv("2015StreetTreesCensus_TREES.csv")
    return trees


def alive_tree(trees):
    # select only alive tree samples
    trees_alive = trees[trees.status == 'Alive'].copy()

    # aggregate health = "Good" of "Fair" to "Good"
    trees_alive["health"] = ["Good" if i == "Fair" else i for i in trees_alive["health"]]

    # drop row with NA in health
    trees_alive.dropna(inplace=True)

    # 

    print(f"Number of NAs: {trees_alive.isna().sum().sum()}. Shape of cleaned dataset: {trees_alive.shape}")

    return trees_alive

