from turtle import down
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
HOME = (
    os.getenv("DOWNLOAD_HOME")
    if os.getenv("DOWNLOAD_HOME")
    else str(Path.home())
)


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


def get_iwildcam(force_download: bool = False, num_location_groups: int = 10):
    """
    Retrieve and break down the IWildCam dataset along the time and location
    (camera ID) domain types.

    Args:
        num_location_groups: how many values there should be for the "location"
            domain type (e.g., if 10, then allocate the 300+ camera IDs into 10
            groups)
    """
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX)
    raw_dataset = get_dataset(
        dataset="iwildcam", download=True, root_dir=download_path
    )

    df = pd.read_csv(
        os.path.join(download_path, "iwildcam_v2.0", "metadata.csv")
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    location_idx = raw_dataset.metadata_fields.index("location")
    time_idx = raw_dataset.metadata_fields.index("month")

    # greedily solve partitioning problem so that camera groups are
    # of roughly equal size
    location_count = raw_dataset.metadata_array[:, location_idx].bincount()
    location_group_map = {}
    location_group_sizes = np.zeros(num_location_groups)

    # assign camera to smallest camera group (by number of examples)
    for location in location_count.argsort(descending=True):
        smallest_group_index = location_group_sizes.argmin().item()
        location_group_map[location.item()] = smallest_group_index
        location_group_sizes[smallest_group_index] += location_count[
            location
        ].item()

    location_matrix = np.zeros((len(df), num_location_groups))
    time_matrix = np.zeros((len(df), df.datetime.dt.month.max() + 1))

    for idx in range(len(raw_dataset.metadata_array)):
        metadata = raw_dataset.metadata_array[idx]
        location_matrix[idx][
            location_group_map[metadata[location_idx].item()]
        ] = 1
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


def get_zillow(force_download: bool = False):
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX, "zillow")
    file_paths = [
        os.path.join(download_path, "Metro_mlp_uc_sfrcondo_week.csv"),
        os.path.join(
            download_path, "Metro_median_sale_price_uc_sfrcondo_week.csv"
        ),
    ]

    if (
        force_download
        or not os.path.exists(file_paths[0])
        or not os.path.exists(file_paths[1])
    ):
        logging.info("Downloading Zillow data")
        list_df = pd.read_csv(
            "https://files.zillowstatic.com/research/public_csvs"
            + "/mlp/Metro_mlp_uc_sfrcondo_week.csv"
        )
        sale_df = pd.read_csv(
            "https://files.zillowstatic.com/research/public_csvs/"
            + "median_sale_price/Metro_median_sale_price_uc_sfrcondo_week.csv"
        )
        os.makedirs(download_path, exist_ok=True)
        list_df.to_csv(file_paths[0], index=False)
        sale_df.to_csv(file_paths[1], index=False)

    list_df = pd.read_csv(file_paths[0])
    sale_df = pd.read_csv(file_paths[1])

    # Clean and transform dataframes (unpivot)
    list_date_cols = list_df.columns.to_list()
    sale_date_cols = sale_df.columns.to_list()
    for col in [
        "RegionID",
        "SizeRank",
        "RegionName",
        "RegionType",
        "StateName",
    ]:
        list_date_cols.remove(col)
        sale_date_cols.remove(col)
    list_price_df = list_df.melt(
        id_vars=["RegionID"],
        value_vars=list_date_cols,
        var_name="date",
        value_name="list_price",
    )
    sale_price_df = sale_df.melt(
        id_vars=["RegionID"],
        value_vars=sale_date_cols,
        var_name="date",
        value_name="sale_price",
    )

    list_price_df["date"] = pd.to_datetime(list_price_df["date"])
    sale_price_df["date"] = pd.to_datetime(sale_price_df["date"])

    list_price_df = list_price_df.dropna(subset=["RegionID"])
    sale_price_df = sale_price_df.dropna(subset=["RegionID"])

    # Resample and forward fill values to merge dfs
    ffill_lim = 10
    list_price_df = (
        list_price_df.groupby("RegionID")
        .apply(
            lambda x: x.set_index("date")
            .sort_values(by="date")
            .resample("1W")
            .ffill(limit=ffill_lim)
            .dropna()
            .reset_index()
        )
        .reset_index(drop=True)
    )
    list_price_df["RegionID"] = list_price_df["RegionID"].astype(int)

    sale_price_df = (
        sale_price_df.groupby("RegionID")
        .apply(
            lambda x: x.set_index("date")
            .sort_values(by="date")
            .resample("1W")
            .ffill(limit=ffill_lim)
            .dropna()
            .reset_index()
        )
        .reset_index(drop=True)
    )
    sale_price_df["RegionID"] = sale_price_df["RegionID"].astype(int)

    # Merge dataframes
    raw_merge = sale_price_df.merge(list_price_df, on=["RegionID"], how="left")
    merged = raw_merge.loc[raw_merge["date_x"] >= raw_merge["date_y"]]
    idx = (
        merged.groupby(["RegionID", "date_x"])["date_y"].transform(max)
        == merged["date_y"]
    )
    merged = merged[idx].reset_index(drop=True)
    merged.rename(
        columns={"date_x": "sale_date", "date_y": "list_date"}, inplace=True
    )
    metro_matrix = pd.get_dummies(merged["RegionID"]).astype(int).values

    # Create dataset
    dataset = RollingDataFrame(
        merged,
        ["sale_price", "list_price"],
        "RegionID",
        label_cols=["sale_price"],
        metadata_cols=["RegionID", "sale_date"],
    )

    return dataset, [metro_matrix]


def get_coauthor(force_download: bool = False):
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX, "coauthor")
    folder_path = os.path.join(download_path, "chi2022-coauthor-v1.0")

    if force_download or not os.path.exists(folder_path):
        logging.info("Downloading CoAuthor data")
        res = requests.get(
            "https://cs.stanford.edu/~minalee/zip/chi2022-coauthor-v1.0.zip",
            stream=True,
        )
        with zipfile.ZipFile(io.BytesIO(res.content)) as zip_ref:
            zip_ref.extractall(download_path)

    session_paths = [
        os.path.join(folder_path, path)
        for path in os.listdir(folder_path)
        if path.endswith("jsonl")
    ]


name_to_func = {
    "mnist": get_mnist,
    "iwildcam": get_iwildcam,
    "civilcomments": get_civil_comments,
    "poverty": get_poverty,
    "jeopardy": get_jeopardy,
    "air_quality": get_air_quality,
    "zillow": get_zillow,
    "coauthor": get_coauthor,
}
