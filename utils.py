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

    # df["startDate"] = df["startDate"].dt.strftime("%Y-%m-%d")
    # df["endDate"] = df["endDate"].dt.strftime("%Y-%m-%d")
    # df["creationDate"] = df["creationDate"].dt.strftime("%Y-%m-%d")
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

    # Reset indexes
    # df.reset_index(inplace=True)
    return df

def netflix_api():
    url = "https://unogs-unogs-v1.p.rapidapi.com/search/titles"

    querystring = {"order_by":"date","title":"Wednesday","type":"movie"}

    headers = {
        "X-RapidAPI-Key": "ab7bd550d6msh5810450efa9635ap180e67jsnbf2ac0823c39",
        "X-RapidAPI-Host": "unogs-unogs-v1.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    print(response.text)

def data_visualization(df, sem):
    plt.figure

    plt.plot(df.index, df["num_of_movies_watched"], label="Number of movie watched")
    plt.plot(df.index, df["sleepTime"], label="Sleep Time")

    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.title(sem)

    plt.show()