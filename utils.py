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
    for row in grouped:
        print(grouped[row])
        # print(grouped.index[row])
        # grouped[row] = int(grouped[row]) - 1
    return grouped

def dow_merge(df):
    dow_df = pd.read_csv("dow.csv")
    df = df.merge(dow_df, on=df.index)
    df.drop("date", axis=1, inplace=True)
    df.rename({"key_0": "creationDate"}, axis=1, inplace=True)
    df["creationDate"] = pd.to_datetime(df["creationDate"])
    return df

def sleep_groupby(df):
    # Date format conversion
    df["startDate"] = df["startDate"].dt.strftime("%Y-%m-%d")
    df["endDate"] = df["endDate"].dt.strftime("%Y-%m-%d")
    df["creationDate"] = df["creationDate"].dt.strftime("%Y-%m-%d")

    # Groupby
    df = df.groupby("creationDate")["sleepTime"].sum()
    df = df.to_frame()
    return df

def sem_split(df):
    # Semester split
    fall_22 = df[(df.creationDate >= "2022-08-30") & (df.creationDate <= "2022-11-29")].copy()
    spring_22 = df[(df.creationDate >= "2022-01-12") & (df.creationDate <= "2022-05-05")].copy()

    # Groupby
    fall_22 = fall_22.groupby("creationDate")["sleepTime"].sum()
    spring_22 = spring_22.groupby("creationDate")["sleepTime"].sum()

    return fall_22, spring_22

def netflix_count(nf):
    # Fall 22
    nf_fa22 = nf[(nf.Date >= "2022-08-30") & (nf.Date <= "2022-11-29")].copy()
    grouped_nf_fa22 = nf_fa22.groupby("Date").count()
    nf_fa22.rename({"Title": "numberWatched"}, axis=1, inplace=True)

    # Spring 22
    nf_sp22 = nf[(nf.Date >= "2022-01-12") & (nf.Date <= "2022-05-05")].copy()
    grouped_nf_sp22 = nf_sp22.groupby("Date").count()
    nf_sp22.rename({"Title": "numberWatched"}, axis=1, inplace=True)
    return grouped_nf_fa22, grouped_nf_sp22

def health_nf_merge(fall_22, spring_22, nf_fa22, nf_sp22):
    # Fall 22
    fall_22 = fall_22.merge(nf_fa22, how="outer", right_index=True, left_index=True)
    fall_22["sleepTime"] = fall_22["sleepTime"] / timedelta(hours=1)

    # Spring 22
    spring_22 = spring_22.merge(nf_sp22, how="outer", right_index=True, left_index=True)
    spring_22["sleepTime"] = spring_22["sleepTime"] / timedelta(hours=1)
    return fall_22, spring_22

def data_visualization(df, sem):
    plt.figure

    plt.plot(df.index, df["num_of_movies_watched"], label="Number of movie watched")
    plt.plot(df.index, df["sleepTime"], label="Sleep Time")

    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.title(sem)

    plt.show()