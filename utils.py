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

def sleep_filtering(df):
    df = df[df.type == "HKCategoryTypeIdentifierSleepAnalysis"]
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

def sem_split(df):
    # Date format conversion
    df["startDate"] = df["startDate"].dt.strftime("%Y-%m-%d")
    df["endDate"] = df["endDate"].dt.strftime("%Y-%m-%d")
    df["creationDate"] = df["creationDate"].dt.strftime("%Y-%m-%d")

    # Semester split
    fall_22 = df[(df.creationDate >= "2022-08-30") & (df.creationDate <= "2022-11-29")]
    spring_22 = df[(df.creationDate >= "2022-01-12") & (df.creationDate <= "2022-05-05")]

    # Groupby
    fall_22 = fall_22.groupby("creationDate").sum()
    # fa22_grouped_by_creationDate = fall_22.groupby("creationDate").sum()
    # sp22_grouped_by_creationDate = spring_22.groupby("creationDate").sum()
    # fall_22 = pd.concat([fall_22, fa22_grouped_by_creationDate], axis=1, ignore_index=False)
    # # df.rename({0: "sleepTime"}, axis=1, inplace=True)
    # spring_22 = pd.concat([spring_22, sp22_grouped_by_creationDate], axis=1, ignore_index=False)

    return fall_22, spring_22

def dow_merge(fall_22, spring_22):
    # Fall 22
    date_fa22 = pd.read_csv("fall_22_dow.csv", index_col=0)
    fall_22 = fall_22.merge(date_fa22, on=fall_22.index)
    fall_22.set_index("key_0", inplace=True)
    fall_22.index = pd.to_datetime(fall_22.index)
    fall_22.rename({"key_0": "creationDate"}, axis=1, inplace=True)

    # Spring 22
    date_sp22 = pd.read_csv("spring_22_dow.csv", index_col=0)
    spring_22 = spring_22.merge(date_sp22, on=spring_22.index)
    spring_22.set_index("key_0", inplace=True)
    spring_22.index = pd.to_datetime(spring_22.index)
    spring_22.rename({"key_0": "creationDate"}, axis=1, inplace=True)
    return fall_22, spring_22

def data_visualization(df, sem):
    plt.figure

    plt.plot(df.index, df["num_of_movies_watched"], label="Number of movie watched")
    plt.plot(df.index, df["sleepTime"], label="Sleep Time")

    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.title(sem)

    plt.show()