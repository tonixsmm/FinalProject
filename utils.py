# Tony Nguyen
# CPSC 222 01
# Dr. Gina Sprint
# December 13th, 2022
# This file contains functions needed to use in ProjectNotebook.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def load_apple():
    # This function loads Apple Health Dataset into a DataFrame and converts date attributes to the right data type
    # Parameter: void
    # Return: a DataFrame

    # Load file
    df = pd.read_csv("export_converted.csv", low_memory=False)
    df.rename(columns={"Unnamed: 0":"originalIndex"}, inplace=True)

    # Convert format
    df["startDate"] = pd.to_datetime(df["startDate"])
    df["endDate"] = pd.to_datetime(df["endDate"])
    df["creationDate"] = pd.to_datetime(df["creationDate"])

    return df

def load_netflix():
    # This function loads Netflix Viewing History dataset into a DataFrame, converts date attributes to the right data type, and sorts it ascendingly by date
    # Parameter: void
    # Return: a DataFrame

    nf = pd.read_csv("NetflixViewingHistory.csv")
    nf["Date"] = pd.to_datetime(nf["Date"])
    nf.sort_values(by="Date", inplace=True, ignore_index=True)

    return nf

def sleep_filtering(df):
    # This function filters which instance is sleep data in "type" attribute
    # Parameter: a DataFramme
    # Return: a DataFrame

    df = df[df.type == "HKCategoryTypeIdentifierSleepAnalysis"].copy()

    return df

def sleep_cleaning(df):
    # This function cleans the data and resets the DataFrame index
    # Parameter: a DataFrame
    # Return: a DataFrame

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
    # This function calculates sleep time and concatenates it to the DataFrame
    # Parameter: a DataFrame
    # Return: a DataFrame
    
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
    # This function aggregates data and calculates awaken time 
    # Parameter: a DataFrame
    # Return: a DataFrame

    # Aggregation
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
    # This function merges the day of week with the DataFrame
    # Parameter: a DataFrame
    # Return: a DataFrame

    # Load and Merge
    dow_df = pd.read_csv("dow.csv")
    df = df.merge(dow_df, on=df.index)

    # Formatting
    df.drop("date", axis=1, inplace=True)
    df.rename({"key_0": "creationDate"}, axis=1, inplace=True)
    df["creationDate"] = pd.to_datetime(df["creationDate"])

    return df

def sleep_groupby(df, grouped):
    # This function aggregates data to find sleep time per night and concatenates it to the DataFrame
    # Parameter: a DataFrame
    # Return: a DataFrame

    # Groupby
    df = df.groupby("creationDate")["sleepTime"].sum()
    df = df.to_frame()

    # Concatenate to the same DataFrame
    df = pd.concat([df, grouped], axis=1, ignore_index=False)
    df.rename({"creationDate": "numberAwake"}, axis=1, inplace=True)

    return df

def sem_split(df):
    # This function splits the data to two groups, Fall 2022 and Spring 2022
    # Parameter: a DataFrame
    # Return: two DataFrames

    # Semester split
    fall_22 = df[(df.creationDate >= "2022-08-30") & (df.creationDate <= "2022-11-29")].copy()
    spring_22 = df[(df.creationDate >= "2022-01-12") & (df.creationDate <= "2022-05-05")].copy()

    # Reset index
    fall_22.reset_index(inplace=True)
    spring_22.reset_index(inplace=True)

    return fall_22, spring_22

def netflix_count(nf):
    # This function counts the number of Netflix shows/movies watched per day
    # Parameter: a DataFrame
    # Return: a DataFrame

    nf = nf.groupby("Date").count()
    nf.rename({"Title": "numberWatched"}, axis=1, inplace=True)

    return nf

def sleep_nf_merge(df, nf):
    # This function merges Apple Health Sleep Data with Netflix Watching History, assigns missing value with 0, and convert time format
    # Parameter: two DataFrame
    # Return: a DataFrame

    # Merge two DataFrame
    nf.rename({"Date":"creationDate"}, axis=1, inplace=True)
    df = df.merge(nf, how="outer", on="creationDate")

    # Assign NaN with 0
    df["numberWatched"] = df["numberWatched"].fillna(0)

    # Convert sleepTime to hour format
    df["sleepTime"] = df["sleepTime"] / timedelta(hours=1)

    return df

def netflix_filtering(df, nf):
    # This function drops a Netflix data's instance if its date is not available in the Apple one
    # Parameter: two DataFrame
    # Return: a DataFrame

    # Check if an instance should be removed
    remove = []
    nf.reset_index(inplace=True)
    for row in range(len(nf.index)):
        count = 0
        for item in range(len(df)):
            if nf.iloc[row, 0] != df.iloc[item, 0]:
                count += 1
        if count == len(df):
            remove.append(row)
    
    # Drop the removed
    nf.drop(remove, axis=0, inplace=True)

    return nf

def data_visualization(fall_22, spring_22, elem):
    # This function visualizes data using bar chart
    # Parameter: two DataFrame and a shared attribute name
    # Return: void

    # Data Preparation
    fa22_grouped_by_dow = fall_22.groupby("dayOfWeek").mean()
    sp22_grouped_by_dow = spring_22.groupby("dayOfWeek").mean()

    # Calculate the horizontal axis label position
    x = np.arange(7)  # the label locations
    width = 0.25 # the width of the bars

    # Plotting
    fig, ax = plt.subplots()
    fall_22 = ax.bar(x - width/2, fa22_grouped_by_dow[elem], width,label="Fall 22")
    return_visit = ax.bar(x + width/2, sp22_grouped_by_dow[elem], width, label="Spring 22")

    # Formatting
    fig.set_figwidth(16)
    fig.set_figheight(9)
    ylabel = "Mean " + elem
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Day of Week")

    ax.set_xticks(x, fa22_grouped_by_dow.index, rotation=45, ha="right")
    ax.legend(title="Semester")

    title = "Mean " + elem + " per Semesters"
    plt.title(title)

    plt.show()

def hypo_2s_1t(test1, test2, alpha, t_critical):
    # This function performs a two-sampled, left-tailed, independent hypothesis test
    # Parameter: two data Series, level of significant, and t_critical value
    # Return: a void

    # Statistic calculation
    t_computed, pval = stats.ttest_ind(test1, test2)
    print("t_computed is:", t_computed)
    print("p_value is:", pval / 2)

    # Condition check
    alpha = alpha
    t_critical = t_critical
    if (pval / 2) < alpha and t_computed < t_critical:
        print("Reject H0")
    elif (pval / 2) > alpha and t_computed >= t_critical:
        print("Do not reject H0")
    else:
        print("Conflicting result")

def ml_preprocessing(df):
    # This function prepares the data to build machine learning model
    # Parameter: a DataFrame
    # Return: a feature matrix DataFrame and a class vector Series

    # Add another attribute to the DataFrame
    week = []
    for row in range(len(df)):
        if df.iloc[row, 3] == "Saturday" or df.iloc[row, 3] == "Sunday":
            week.append("No")
        else:
            week.append("Yes")
    
    week = pd.Series(week, dtype=object)
    df = pd.concat([df, week], axis=1, ignore_index=False)
    df.rename({0: "isWeekday"}, axis=1, inplace=True)

    df.drop("creationDate", axis=1, inplace=True)

    # Label encoding
    le = LabelEncoder()
    for item in df.columns:
        if item == "dayOfWeek" or item == "isWeekday":
            le.fit(df[item])
            df[item] = le.transform(df[item])

    # Split the data to form a feature matrix and a class vector
    y = df["dayOfWeek"]
    X = df.drop("dayOfWeek", axis=1)

    return X, y

def scale_split(X, y):
    # This function normalizes the feature matrix and performs a split to form a training set and a testing set
    # Parameter: a DataFrame and a Series
    # Return: two DataFrames and two Series

    # Normalization
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_normalized = scaler.transform(X)

    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.25, random_state=0, stratify=y)

    return X_train, X_test, y_train, y_test

def kNN_class(X_train, X_test, y_train, y_test):
    # This function builds the kNeighborsClassifier machine learning model and finds the best number of neighbors
    # Parameter: two DataFrames and two Series
    # Return: void

    acc_test = 0
    position = 0
    for i in range(149):
        knn_clf = KNeighborsClassifier(n_neighbors=i + 1)
        knn_clf.fit(X_train, y_train)
        y_pred = knn_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if acc > acc_test:
            acc_test = acc
            position = i
    print("Best accuracy:", acc_test, "Number of Neighbors:", position + 1)

def tree_class(X_train, X_test, y_train, y_test):
    # This function builds the DecisionTreeClassifier machine learning model
    # Parameter: two DataFrames and two Series
    # Return: void

    tree_clf = DecisionTreeClassifier(random_state=0)
    tree_clf.fit(X_train, y_train)
    acc = tree_clf.score(X_test, y_test)
    print("Accuracy:", acc)