from dotenv import load_dotenv
from pathlib import Path
from torchvision import datasets
from streams.utils import FullDataset, SimpleDataset, RollingDataFrame
from wilds import get_dataset

import io
import logging
import numpy as np
import os
import pandas as pd
import requests
import subprocess
import torch
import torchvision.transforms as transforms
import typing
import zipfile

load_dotenv()
DOWNLOAD_PREFIX = (
    os.getenv("DOWNLOAD_PREFIX")
    if os.getenv("DOWNLOAD_PREFIX")
    else "streams_data"
)
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
HOME = str(Path.home())


def get_mnist(force_download: bool = False):
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX, "mnist")
    mnist_train = datasets.MNIST(download_path, train=True, download=True)
    mnist_test = datasets.MNIST(download_path, train=False, download=True)

    # Concate train and test
    dataset = mnist_train + mnist_test

    # There will be 1 domain and 10 values (labels)
    domain_matrix = np.zeros((len(dataset), 10))
    for idx, elem in enumerate(dataset):
        domain_matrix[idx][elem[1]] = 1

    return dataset, [domain_matrix]


def get_iwildcam(force_download: bool = False):
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX)
    raw_dataset = get_dataset(
        dataset="iwildcam", download=True, root_dir=download_path
    )

    df = pd.read_csv(
        os.path.join(download_path, "iwildcam_v2.0", "metadata.csv")
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    location_matrix = np.zeros((len(df), df.location_remapped.max() + 1))
    time_matrix = np.zeros((len(df), df.datetime.dt.month.max() + 1))
    location_idx = raw_dataset.metadata_fields.index("location")
    time_idx = raw_dataset.metadata_fields.index("month")

    for idx in range(len(raw_dataset.metadata_array)):
        metadata = raw_dataset.metadata_array[idx]
        location_matrix[idx][metadata[location_idx]] = 1
        time_matrix[idx][metadata[time_idx]] = 1

    dataset = FullDataset(
        raw_dataset,
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )

    return dataset, [location_matrix, time_matrix]


def get_civil_comments(force_download: bool = False):
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX)
    raw_dataset = get_dataset(
        dataset="civilcomments", download=True, root_dir=download_path
    )
    df = pd.read_csv(
        os.path.join(
            download_path, "civilcomments_v1.0", "all_data_with_identities.csv"
        )
    )

    all_cols = {
        "gender_cols": ["male", "female", "transgender", "other_gender"],
        "sexuality_cols": [
            "heterosexual",
            "homosexual_gay_or_lesbian",
            "bisexual",
            "other_sexual_orientation",
        ],
        "religion_cols": [
            "christian",
            "jewish",
            "muslim",
            "hindu",
            "buddhist",
            "atheist",
            "other_religion",
        ],
        "race_cols": [
            "black",
            "white",
            "asian",
            "latino",
            "other_race_or_ethnicity",
        ],
        "disability_cols": [
            "physical_disability",
            "intellectual_or_learning_disability",
            "psychiatric_or_mental_illness",
            "other_disability",
        ],
    }

    matrices = []
    for _, cols in all_cols.items():
        matrix = (df[cols] >= 0.5).astype(int).values
        last_col = (matrix.sum(axis=1) == 0).astype(int).reshape(-1, 1)
        matrix = np.hstack([matrix, last_col])
        matrices.append(matrix)

    publication_id_matrix = (
        pd.get_dummies(df.publication_id).astype(int).values
    )
    matrices.append(publication_id_matrix)

    dataset = FullDataset(raw_dataset)

    return dataset, matrices


def get_poverty(force_download: bool = False):
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX)
    raw_dataset = get_dataset(
        dataset="poverty", download=True, root_dir=download_path
    )
    df = pd.read_csv(
        os.path.join(download_path, "poverty_v1.1", "dhs_metadata.csv")
    )

    urban_matrix = pd.get_dummies(df.urban).astype(int).values
    country_matrix = pd.get_dummies(df.country).astype(int).values
    dataset = FullDataset(raw_dataset)

    return dataset, [urban_matrix, country_matrix]


def get_jeopardy(force_download: bool = False):
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX, "jeopardy")
    command = (
        "kaggle datasets download -d "
        + f"tunguz/200000-jeopardy-questions -p {download_path}"
    )
    command += " --force" if force_download else ""

    subprocess.run(command, shell=True)

    if not os.path.exists(os.path.join(download_path, "JEOPARDY_CSV.csv")):
        zip_path = os.path.join(download_path, "200000-jeopardy-questions.zip")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(download_path)

    df = pd.read_csv(os.path.join(download_path, "JEOPARDY_CSV.csv"))
    df.replace(to_replace="None", value=np.nan, inplace=True)
    df.rename(
        {
            " Value": "Value",
            " Category": "Category",
            " Question": "Question",
            " Answer": "Answer",
        },
        axis=1,
        inplace=True,
    )

    df = df.dropna(subset=["Value", "Category"])

    df["Category"] = df["Category"].astype(str).str.strip().str.lower()
    df["Value"] = df["Value"].astype(str).str.strip().str.lower()
    df["Value"] = (
        df["Value"].str.replace("$", "").str.replace(",", "").astype(float)
    )

    value_matrix = pd.get_dummies(df["Value"]).astype(int).values
    category_matrix = pd.get_dummies(df["Category"]).astype(int).values

    dataset = SimpleDataset(df["Question"].values, df["Answer"].values)
    return dataset, [value_matrix]  # TODO add category matrix


def get_air_quality(force_download: bool = False):
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX, "air_quality")
    folder_path = os.path.join(download_path, "PRSA_Data_20130301-20170228")

    if force_download or not os.path.exists(folder_path):
        logging.info("Downloading air quality data")
        res = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            + "/00501/PRSA2017_Data_20130301-20170228.zip",
            stream=True,
        )
        with zipfile.ZipFile(io.BytesIO(res.content)) as zip_ref:
            zip_ref.extractall(download_path)

    dfs = []
    for filename in os.listdir(folder_path):
        dfs.append(pd.read_csv(os.path.join(folder_path, filename)))

    df = pd.concat(dfs).reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["hour"] = df["hour"].astype(int)

    # Make one of the sensors int valued
    wds = df.wd.unique().tolist()
    df["wd_raw"] = df["wd"]
    df["wd"] = df["wd"].apply(lambda x: wds.index(x))

    # Create station matrix
    station_matrix = pd.get_dummies(df["station"]).astype(int).values

    # Create dataset
    sensor_cols = [
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",
        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
        "wd",
        "WSPM",
    ]
    dataset = RollingDataFrame(df, sensor_cols, "station")
    return dataset, [station_matrix]


name_to_func = {
    "mnist": get_mnist,
    "iwildcam": get_iwildcam,
    "civilcomments": get_civil_comments,
    "poverty": get_poverty,
    "jeopardy": get_jeopardy,
    "air_quality": get_air_quality,
}
