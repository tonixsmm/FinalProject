import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy import stats

def load_apple():
    # Load file
    df = pd.read_csv("export_converted.csv")

    df.rename(columns={"Unnamed: 0":"originalIndex"}, inplace=True)

    # Convert format
    df["startDate"] = pd.to_datetime(df["startDate"])
    df["endDate"] = pd.to_datetime(df["endDate"])
    df["creationDate"] = pd.to_datetime(df["creationDate"])

    return df

def load_netflix():
    nf = pd.read_csv("NetflixViewingHistory.csv")
    nf["Date"] = pd.to_datetime(nf["Date"])
    nf.sort_values(by="Date", inplace=True, ignore_index=True)

    return nf

def sleep_filtering(df):
    df = df[df.type == "HKCategoryTypeIdentifierSleepAnalysis"].copy()

    return df

def sleep_cleaning(df):
    # Drop columns
    df.drop("type", axis=1, inplace=True)
    df.drop("sourceName", axis=1, inplace=True)
    df.drop("sourceVersion", axis=1, inplace=True)
    df.drop("unit", axis=1, inplace=True)
    df.drop("value", axis=1, inplace=True)
    df.drop("device", axis=1, inplace=True)
    df.drop("MetadataEntry", axis=1, inplace=True)
    df.drop("HeartRateVariabilityMetadataList", axis=1, inplace=True)
    df.drop("originalIndex", axis=1, inplace=True)

    # Reset indexes
    df.reset_index(inplace=True)

    return df

def time_calculation(df):
    # Substraction
    sleepTime = []
    for item in range(len(df)):
        sleepTime.append(df.iloc[item, 3] - df.iloc[item, 2])
    sleepTime = pd.Series(sleepTime)

    # Concatenate to the DataFrame
    df = pd.concat([df, sleepTime], axis=1, ignore_index=False)
    df.rename({0: "sleepTime"}, axis=1, inplace=True)

    return sleepTime, df

def awaken_time(df):
    df["creationDate"] = df["creationDate"].dt.strftime("%Y-%m-%d")
    grouped = df["creationDate"].value_counts()

    # Substract its value for 1 to find the actual awaken time
    for row in grouped.index:
        grouped.loc[row] = grouped.loc[row] - 1

    # Manual cleaning
    grouped["2022-10-23"] = 0
    grouped["2022-09-14"] = 0
    grouped["2022-10-18"] = 2
    grouped["2022-10-22"] = 0
    grouped["2022-09-17"] = 2
    grouped["2022-07-04"] = 0
    grouped["2021-12-23"] = 0
    grouped["2021-09-27"] = 0
    grouped["2021-12-20"] = 1
    grouped["2022-09-12"] = 0
    grouped["2022-11-02"] = 0
    grouped["2021-10-20"] = 0
    grouped["2022-02-07"] = 2
    grouped["2022-10-30"] = 1
    grouped["2021-12-24"] = 2
    grouped["2022-01-09"] = 1
    grouped["2021-12-08"] = 2
    grouped["2021-12-31"] = 1
    grouped["2022-09-22"] = 0
    grouped["2022-09-27"] = 5

    return grouped

def dow_merge(df):
    dow_df = pd.read_csv("dow.csv")
    df = df.merge(dow_df, on=df.index)
    df.drop("date", axis=1, inplace=True)
    df.rename({"key_0": "creationDate"}, axis=1, inplace=True)
    df["creationDate"] = pd.to_datetime(df["creationDate"])

    return df

def sleep_groupby(df, grouped):
    # Groupby
    df = df.groupby("creationDate")["sleepTime"].sum()
    df = df.to_frame()

    # Concatenate to the same DataFrame
    df = pd.concat([df, grouped], axis=1, ignore_index=False)
    df.rename({"creationDate": "numberAwake"}, axis=1, inplace=True)

    return df

def sem_split(df):
    # Semester split
    fall_22 = df[(df.creationDate >= "2022-08-30") & (df.creationDate <= "2022-11-29")].copy()
    spring_22 = df[(df.creationDate >= "2022-01-12") & (df.creationDate <= "2022-05-05")].copy()

    # Reset index
    fall_22.reset_index(inplace=True)
    spring_22.reset_index(inplace=True)

    return fall_22, spring_22

def netflix_count(nf):
    nf = nf.groupby("Date").count()
    nf.rename({"Title": "numberWatched"}, axis=1, inplace=True)

    return nf

def health_nf_merge(fall_22, spring_22, nf_fa22, nf_sp22):
    # Fall 22
    fall_22 = fall_22.merge(nf_fa22, how="outer", right_index=True, left_index=True)
    fall_22["sleepTime"] = fall_22["sleepTime"] / timedelta(hours=1)

    # Spring 22
    spring_22 = spring_22.merge(nf_sp22, how="outer", right_index=True, left_index=True)
    spring_22["sleepTime"] = spring_22["sleepTime"] / timedelta(hours=1)

    return fall_22, spring_22

def sleep_nf_merge(df, nf):
    # Merge two DataFrame
    nf.rename({"Date":"creationDate"}, axis=1, inplace=True)
    df = df.merge(nf, how="outer", on="creationDate")

    # Assign NaN with 0
    df["numberWatched"] = df["numberWatched"].fillna(0)

    # Convert sleepTime to hour format
    df["sleepTime"] = df["sleepTime"] / timedelta(hours=1)

    return df

def netflix_filtering(df, nf):
    remove = []
    nf.reset_index(inplace=True)
    for row in range(len(nf.index)):
        count = 0
        for item in range(len(df)):
            if nf.iloc[row, 0] != df.iloc[item, 0]:
                count += 1
        if count == len(df):
            remove.append(row)
    
    nf.drop(remove, axis=0, inplace=True)

    return nf

def data_visualization(fall_22, spring_22, elem):
    fa22_grouped_by_dow = fall_22.groupby("dayOfWeek").mean()
    sp22_grouped_by_dow = spring_22.groupby("dayOfWeek").mean()

    x = np.arange(7)  # the label locations
    width = 0.25 # the width of the bars

    fig, ax = plt.subplots()
    fall_22 = ax.bar(x - width/2, fa22_grouped_by_dow[elem], width,label="Fall 22")
    return_visit = ax.bar(x + width/2, sp22_grouped_by_dow[elem], width, label="Spring 22")

    fig.set_figwidth(16)
    fig.set_figheight(9)
    ylabel = "Mean " + elem
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Day of Week")

    ax.set_xticks(x, fa22_grouped_by_dow.index, rotation=45, ha="right")
    ax.legend(title="Semester")

    title = "Mean " + elem + " and Semesters"
    plt.title(title)

    plt.show()

def hypo_2s_1t(test1, test2, alpha, t_critical):
    t_computed, pval = stats.ttest_ind(test1, test2)
    print("t_computed is:", t_computed)
    print("p_value is:", pval / 2)

    alpha = alpha
    t_critical = t_critical
    if (pval / 2) < alpha and t_computed < t_critical:
        print("Reject H0")
    elif (pval / 2) > alpha and t_computed >= t_critical:
        print("Do not reject H0")
    else:
        print("Conflicting result")