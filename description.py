import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# function get datasets of common tree species
def get_common_tree(df):
    """get datasets contain common trees i.e. over 10k count"""
    # get tree names which amount over 10k
    tree_counts = df.spc_common.value_counts()
    trees_over_10k = tree_counts[tree_counts >= 10000].index

    # filter them out from original dataset
    trees_alive_10k = df[df["spc_common"].isin(trees_over_10k)].copy()

    return trees_alive_10k

# function that extract contengency table, and plot as heatmap
def contingency_heatmap(df, iv_list, label):
    # import gridspec for subplots
    from matplotlib import gridspec

    # set the figure
    n_iv = len(iv_list)
    gs = gridspec.GridSpec(np.ceil(n_iv/4).astype(int), 4)
    fig = plt.figure(figsize = (20, 10))
    
    # iterate over iv_list
    for index, iv in enumerate(iv_list):
        # get contingency table
        contingency = pd.crosstab(df[iv], df[label], normalize='columns')
        # return heatmap
        ax = fig.add_subplot(gs[index])
        ax = sns.heatmap(contingency, annot=True, cmap="Greens", cbar= False)
        
    fig.tight_layout(h_pad=2.5, w_pad = 2.5)
     
    return fig

def prop_z_test(df, specie, alpha=0.01):
    """get confusion matrix and perfom proportion z test"""

    from statsmodels.stats.proportion import proportions_ztest

    df[specie] = df["spc_common"].apply(lambda x: x == specie)
    crosstab = pd.crosstab(df.health, df[specie], margins=True)

    nobs = np.array([crosstab.values[2,1], crosstab.values[2,0]])
    count = np.array([ crosstab.values[1,1], crosstab.values[1,0]])
    stat, pval = proportions_ztest(count, nobs)
    print(crosstab, "\n")
    print(f"Z-score = {stat:.3}, p = {pval:.3}.")
    if pval >= alpha: 
        print(f"{pval:.3} > {alpha}, the null hypothesis is not rejected.")
    else:
        print(f"{pval:.3} < {alpha}, the null hypothesis is rejected.")
    return 

def independence_test(df, target, alpha=0.01):
    """perform chi-square test of independece"""

    # make contingency table of brnch_other vs health
    contingency = pd.crosstab(df[target], df.health)
    print(contingency, "\n")

    # set alpha = 0.01, perform chi-square test of independece
    chi, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"Chi-square = {chi:.3}, p = {p:.3}.")
    if p >= alpha:
        print(f"{p:.3} > {alpha}, the null hypothesis is not rejected.")
    else:
        print(f"{p:.3} < {alpha}, the null hypothesis is rejected.")

    return 