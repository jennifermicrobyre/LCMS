
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats

def inputs():
    filename = 'ProFinderExport_11-02-2023_pos_reprocess.csv'
    file_loc = '/Users/jennifer/Documents/Analysis/LCMS/Initial_Runs/'

    name_mass_column = 'Mass (avg)'

    # experiments to compare
    exp1 = ['[Area] TSB_pos-r001d', '[Area] TSB_pos-r002d', '[Area] TSB_pos-r003']
    exp2 = ['[Area] TSB_339_pos-r001', '[Area] TSB_339_pos-r002', '[Area] TSB_339_pos-r003']

    num_data_points_required = 3

    return filename, file_loc, name_mass_column, exp1, exp2, num_data_points_required

def get_areas(df, exp1, exp2):
    area1 = []
    area2 = []

    for i in range(len(exp1)):
        if pd.isna(df[exp1[i]].iloc[0]) == False:
            area1.append(df[exp1[i]].iloc[0])

    for i in range(len(exp2)):
        if pd.isna(df[exp2[i]].iloc[0]) == False:
            area2.append(df[exp2[i]].iloc[0])

    area1 = np.array(area1)
    area2 = np.array(area2)

    return area1, area2

def get_p_values(area1, area2, num_data_points_required):
    if area1.size < num_data_points_required or area2.size < num_data_points_required:
        pvalue = np.NaN
    else:
        t, pvalue = ttest_ind_from_stats(mean1=np.mean(area1), std1=np.std(area1), nobs1=area1.size,
                                         mean2=np.mean(area2), std2=np.std(area2), nobs2=area2.size)

    return pvalue

def get_fold_change(area1, area2, num_data_points_required):
    if area1.size < num_data_points_required or area2.size < num_data_points_required:
        fold_change = np.NaN
    else:
        fold_change = np.mean(area1)/np.mean(area2)

    return fold_change

def get_p_and_fold_change(df, name_mass_column, exp1, exp2, num_data_points_required):
    unique_masses = pd.unique(df[name_mass_column])

    p = np.zeros(len(unique_masses))
    fold_change = np.zeros(len(unique_masses))

    for i in range(len(unique_masses)):
        subset = df.loc[(df[name_mass_column] == unique_masses[i])]
        area1, area2 = get_areas(subset, exp1, exp2)
        p[i] = get_p_values(area1, area2, num_data_points_required)
        fold_change[i] = get_fold_change(area1, area2, num_data_points_required)

    return p, fold_change

def plot(p, fold_change):
    plt.scatter(np.log2(fold_change), -np.log10(p), marker='.')
    plt.ylabel('-log10 p-Value')
    plt.xlabel('log2 Fold Change')
    plt.axhline(y=1.3, color="black", linestyle="--")
    plt.show()

def main():
    filename, file_loc, name_mass_column, exp1, exp2, num_data_points_required = inputs()
    df = pd.read_csv(file_loc + filename)
    p, fold_change = get_p_and_fold_change(df, name_mass_column, exp1, exp2, num_data_points_required)
    plot(p, fold_change)

main()
