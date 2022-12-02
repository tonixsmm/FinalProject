import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

def load_xml(filename):
    df = pd.read_xml(filename, xpath="Record")
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