import io
import json
import logging
import os
import subprocess
import tarfile
import typing
import zipfile
from pathlib import Path
from turtle import down
from unicodedata import category

import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
from nuimages import NuImages
from torchvision import datasets
from wilds import get_dataset

from streams.utils import (FullDataset, NuImagesDataset, RollingDataFrame,
                           SimpleDataset, get_prompts_and_completions)

load_dotenv()
DOWNLOAD_PREFIX = (
    os.getenv("DOWNLOAD_PREFIX") if os.getenv("DOWNLOAD_PREFIX") else "streams_data"
)
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
HOME = os.getenv("DOWNLOAD_HOME") if os.getenv("DOWNLOAD_HOME") else str(Path.home())


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

    return dataset, [domain_matrix], None


def get_iwildcam(
    force_download: bool = False, num_location_groups: int = 10
) -> typing.Tuple[torch.utils.data.Dataset, typing.List[np.ndarray], np.ndarray]:
    """
    Retrieve and break down the IWildCam dataset along the time and location
    (camera ID) domain types.

    Args:
        num_location_groups: how many values there should be for the "location"
            domain type (e.g., if 10, then allocate the 300+ camera IDs into 10
            groups)

    Returns:
        dataset: utils.FullDataset
        domain_matrices: list of np.ndarray
        time_periods: np.ndarray (or None, if time is not a domain)
    """
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX)
    raw_dataset = get_dataset(dataset="iwildcam", download=True, root_dir=download_path)

    df = pd.read_csv(os.path.join(download_path, "iwildcam_v2.0", "metadata.csv"))
    df["datetime"] = pd.to_datetime(df["datetime"])
    location_idx = raw_dataset.metadata_fields.index("location")

    # greedily solve partitioning problem so that camera groups are
    # of roughly equal size
    location_count = raw_dataset.metadata_array[:, location_idx].bincount()
    location_group_map = {}
    location_group_sizes = np.zeros(num_location_groups)

    # assign camera to smallest camera group (by number of examples)
    for location in location_count.argsort(descending=True):
        smallest_group_index = location_group_sizes.argmin().item()
        location_group_map[location.item()] = smallest_group_index
        location_group_sizes[smallest_group_index] += location_count[location].item()

    location_matrix = np.zeros((len(df), num_location_groups))

    for idx in range(len(raw_dataset.metadata_array)):
        metadata = raw_dataset.metadata_array[idx]
        location_matrix[idx][location_group_map[metadata[location_idx].item()]] = 1

    dataset = FullDataset(
        raw_dataset,
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )

    time_idx = raw_dataset.metadata_fields.index("month")
    time_periods = raw_dataset.metadata_array[:, time_idx].numpy()

    return dataset, [location_matrix], time_periods


def get_civil_comments(
    force_download: bool = False,
) -> typing.Tuple[torch.utils.data.Dataset, typing.List[np.ndarray], np.ndarray]:
    """Retrieves and preprocesses the Civil Comments dataset. Domains are
    gender, sexuality, race, religion, and disability.

    Args:
        force_download (bool, optional): Defaults to False.

    Returns:
        typing.Tuple[ torch.utils.data.Dataset, typing.List[np.ndarray],
            np.ndarray ]: Dataset, domain matrices of size (num_examples,
            num_domain_vals) for each domain, and time periods
            (if time is a domain)
    """
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

    publication_id_matrix = pd.get_dummies(df.publication_id).astype(int).values
    matrices.append(publication_id_matrix)

    dataset = FullDataset(raw_dataset)

    return dataset, matrices, None


def get_poverty(
    force_download: bool = False,
) -> typing.Tuple[torch.utils.data.Dataset, typing.List[np.ndarray], np.ndarray]:
    """Retrieves and preprocesses the Poverty dataset. Domains are urban
    indicator and country.

    Args:
        force_download (bool, optional): Defaults to False.

    Returns:
        typing.Tuple[ torch.utils.data.Dataset, typing.List[np.ndarray],
            np.ndarray ]: Dataset, domain matrices of size (num_examples,
            num_domain_vals) for each domain, and time periods
            (if time is a domain)
    """
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX)
    raw_dataset = get_dataset(dataset="poverty", download=True, root_dir=download_path)
    df = pd.read_csv(os.path.join(download_path, "poverty_v1.1", "dhs_metadata.csv"))

    urban_matrix = pd.get_dummies(df.urban).astype(int).values
    country_matrix = pd.get_dummies(df.country).astype(int).values
    dataset = FullDataset(raw_dataset)

    return dataset, [urban_matrix, country_matrix], None


def get_jeopardy(
    force_download: bool = False,
) -> typing.Tuple[torch.utils.data.Dataset, typing.List[np.ndarray], np.ndarray]:
    """Retrieves and preprocesses the Jeopardy dataset. Domains are question
    value amount and month of year.

    Args:
        force_download (bool, optional): Defaults to False.

    Returns:
        typing.Tuple[ torch.utils.data.Dataset, typing.List[np.ndarray],
            np.ndarray ]: Dataset, domain matrices of size (num_examples,
            num_domain_vals) for each domain, and time periods
            (if time is a domain)
    """
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
            " Air Date": "Air Date",
        },
        axis=1,
        inplace=True,
    )

    df = df.dropna(subset=["Value", "Category"])

    df["Category"] = df["Category"].astype(str).str.strip().str.lower()
    df["Value"] = df["Value"].astype(str).str.strip().str.lower()
    df["Value"] = df["Value"].str.replace("$", "").str.replace(",", "").astype(float)
    df["Air Date"] = pd.to_datetime(df["Air Date"])

    value_matrix = pd.get_dummies(df["Value"]).astype(int).values
    month_matrix = pd.get_dummies(df["Air Date"].dt.month).astype(int).values

    dataset = SimpleDataset(df, feature_cols=["Question"], label_cols=["Answer"])
    return dataset, [value_matrix, month_matrix], None


def get_air_quality(
    force_download: bool = False,
) -> typing.Tuple[torch.utils.data.Dataset, typing.List[np.ndarray], np.ndarray]:
    """Retrieves and preprocesses the Beijing Air Quality dataset.
    Domain is the station the measurement was taken from.

    Args:
        force_download (bool, optional): Defaults to False.

    Returns:
        typing.Tuple[ torch.utils.data.Dataset, typing.List[np.ndarray],
            np.ndarray ]: Dataset, domain matrices of size (num_examples,
            num_domain_vals) for each domain, and time periods
            (if time is a domain)
    """
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX, "air_quality")
    folder_path = os.path.join(download_path, "PRSA_Data_20130301-20170228")

    if force_download or not os.path.exists(folder_path):
        logging.debug("Downloading air quality data")
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
    timestamps = pd.to_datetime(df[["year", "month", "day", "hour"]])

    # Make one of the sensors int valued
    wds = df.wd.unique().tolist()
    df["wd_raw"] = df["wd"]
    df["wd"] = df["wd"].apply(lambda x: wds.index(x))

    # Create station matrix
    station_matrix = pd.get_dummies(df["station"]).astype(int).values

    # Create time periods array. We use dataframes for speed, otherwise it
    # will take O(10 mins).
    unique_timestamps = sorted(timestamps.unique().tolist())
    timestamp_map = pd.DataFrame(
        {
            "timestamp": unique_timestamps,
            "index": range(0, len(unique_timestamps)),
        }
    )
    timestamp_map["timestamp"] = pd.to_datetime(timestamp_map["timestamp"])
    timestamp_idx = (
        timestamps.to_frame(name="timestamp")
        .merge(timestamp_map, on="timestamp", how="left")["index"]
        .values
    )

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
    return dataset, [station_matrix], timestamp_idx


def get_zillow(
    force_download: bool = False,
) -> typing.Tuple[torch.utils.data.Dataset, typing.List[np.ndarray], np.ndarray]:
    """Retrieves and preprocesses the Zillow dataset. Domain is the
    metro / area.

    Args:
        force_download (bool, optional): Defaults to False.

    Returns:
        typing.Tuple[ torch.utils.data.Dataset, typing.List[np.ndarray],
            np.ndarray ]: Dataset, domain matrices of size (num_examples,
            num_domain_vals) for each domain, and time periods
            (if time is a domain)
    """
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX, "zillow")
    file_paths = [
        os.path.join(download_path, "Metro_mlp_uc_sfrcondo_week.csv"),
        os.path.join(download_path, "Metro_median_sale_price_uc_sfrcondo_week.csv"),
    ]

    if (
        force_download
        or not os.path.exists(file_paths[0])
        or not os.path.exists(file_paths[1])
    ):
        logging.debug("Downloading Zillow data")
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
    merged.rename(columns={"date_x": "sale_date", "date_y": "list_date"}, inplace=True)
    metro_matrix = pd.get_dummies(merged["RegionID"]).astype(int).values
    unique_sale_dates = sorted(merged["sale_date"].unique())
    sale_idx = merged["sale_date"].apply(lambda x: unique_sale_dates.index(x)).values

    # Create dataset
    dataset = RollingDataFrame(
        merged,
        ["sale_price", "list_price"],
        "RegionID",
        label_cols=["sale_price"],
        metadata_cols=["RegionID", "sale_date"],
    )

    return dataset, [metro_matrix], sale_idx


def get_coauthor(
    force_download: bool = False,
) -> typing.Tuple[torch.utils.data.Dataset, typing.List[np.ndarray], np.ndarray]:
    """Retrieves and preprocesses the Coauthor dataset. Domains are worker
    ID and prompt category.

    Args:
        force_download (bool, optional): Defaults to False.

    Returns:
        typing.Tuple[ torch.utils.data.Dataset, typing.List[np.ndarray],
            np.ndarray ]: Dataset, domain matrices of size (num_examples,
            num_domain_vals) for each domain, and time periods
            (if time is a domain)
    """
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX, "coauthor")
    folder_path = os.path.join(download_path, "coauthor-v1.0")

    if force_download or not os.path.exists(folder_path):
        logging.debug("Downloading CoAuthor data")
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
    events = [
        [json.loads(event) for event in open(path, "r")] for path in session_paths
    ]

    dfs = []
    drop_keyword = "DROP_KEYWORD"
    for i, path in enumerate(os.listdir(folder_path)):
        dfs.append(
            get_prompts_and_completions(
                events[i],
                session_id=path,
                drop_keyword=drop_keyword,
            )
        )

    df = pd.concat(dfs)
    df = df[df["next"] != drop_keyword].reset_index(drop=True)

    # Read metadata files for domain values
    def build_sheet_url(doc_id, sheet_id):
        return (
            f"https://docs.google.com/spreadsheets/d/{doc_id}"
            + f"/export?format=csv&gid={sheet_id}"
        )

    doc_id = "1O3EXJm52TQHfFSbzVGZmNIzzdu5ow6IjnOBrGTUY02o"
    creative_sheet_id = "1870708729"
    argumentative_sheet_id = "320516663"

    creative_metadata = pd.read_csv(build_sheet_url(doc_id, creative_sheet_id))
    argumentative_metadata = pd.read_csv(
        build_sheet_url(doc_id, argumentative_sheet_id)
    )

    # Make sure metadata files are well-formed
    assert creative_metadata["session_id"].nunique() == len(creative_metadata)
    assert argumentative_metadata["session_id"].nunique() == len(argumentative_metadata)
    metadata = pd.concat(
        [
            creative_metadata[["worker_id", "session_id", "prompt_code"]],
            argumentative_metadata[["worker_id", "session_id", "prompt_code"]],
        ]
    ).reset_index(drop=True)
    assert metadata["session_id"].nunique() == len(metadata)

    # Merge metadata and df to create domain matrices
    full_df = df.merge(metadata, on="session_id", how="left")
    worker_matrix = pd.get_dummies(full_df["worker_id"]).astype(int).values
    prompt_matrix = pd.get_dummies(full_df["prompt_code"]).astype(int).values

    # Create dataset
    dataset = SimpleDataset(
        full_df,
        feature_cols=["prompt", "current"],
        label_cols=["next"],
        metadata_cols=["worker_id", "prompt_code"],
    )

    return dataset, [worker_matrix, prompt_matrix], None


def get_voxceleb(force_download: bool = False):
    pass


def get_test(
    force_download=False,
) -> typing.Tuple[torch.utils.data.Dataset, typing.List[np.ndarray], np.ndarray]:
    """Testing utility function. Creates a fake dataset.

    Returns:
        typing.Tuple[ torch.utils.data.Dataset, typing.List[np.ndarray],
            np.ndarray ]: Dataset, domain matrices of size (num_examples,
            num_domain_vals) for each domain, and time periods
            (if time is a domain)
    """
    df = pd.DataFrame(
        {
            "feat1": list(range(20)),
            "feat2": list(range(20, 40)),
            "label": [1] * 20,
        }
    )

    matrix = pd.get_dummies(df["feat1"]).astype(int).values
    dataset = SimpleDataset(df, feature_cols=["feat1", "feat2"], label_cols=["label"])

    return dataset, [matrix], None


def get_nuimages(force_download=False):
    """Retrieves and preprocesses the NuImages dataset. Domains are
    modality, location, and vehicle ID.

    Args:
        force_download (bool, optional): Defaults to False.

    Returns:
        typing.Tuple[ torch.utils.data.Dataset, typing.List[np.ndarray],
            np.ndarray ]: Dataset, domain matrices of size (num_examples,
            num_domain_vals) for each domain, and time periods
            (if time is a domain)
    """
    download_path = os.path.join(HOME, DOWNLOAD_PREFIX, "nuscenes")

    # TODO(shreyashankar): Download full dataset instead of mini
    if force_download or not os.path.exists(download_path):
        logging.debug("Downloading nuscenes images data")
        res = requests.get(
            "https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz",
            stream=True,
        )
        with tarfile.open(fileobj=io.BytesIO(res.content)) as ref:
            ref.extractall(download_path)

    nuim = NuImages(
        dataroot=download_path,
        version="v1.0-mini",
        lazy=True,
    )

    # Create domain matrices
    modalities = []
    locations = []
    vehicles = []
    categories = []

    for data in nuim.object_ann:
        categories.append(data["category_token"])
        modalities.append(
            nuim.get(
                "sensor",
                nuim.get(
                    "calibrated_sensor",
                    nuim.get("sample_data", data["sample_data_token"])[
                        "calibrated_sensor_token"
                    ],
                )["sensor_token"],
            )["modality"]
        )
        log = nuim.get(
            "log",
            nuim.get(
                "sample",
                nuim.get("sample_data", data["sample_data_token"])["sample_token"],
            )["log_token"],
        )
        locations.append(log["location"])
        vehicles.append(log["vehicle"])

    modality_matrix = pd.get_dummies(modalities).astype(int).values
    location_matrix = pd.get_dummies(locations).astype(int).values
    vehicle_matrix = pd.get_dummies(vehicles).astype(int).values

    # Create dataset and labels
    distinct_labels = list(np.unique(categories))
    labels = [distinct_labels.index(x) for x in categories]
    dataset = NuImagesDataset(
        nuim,
        labels,
        {"modality": modalities, "location": locations, "vehicle": vehicles},
        download_path,
    )

    return dataset, [modality_matrix, location_matrix, vehicle_matrix], None


name_to_func = {
    "mnist": get_mnist,
    "iwildcam": get_iwildcam,
    "civilcomments": get_civil_comments,
    "poverty": get_poverty,
    "jeopardy": get_jeopardy,
    "airquality": get_air_quality,
    "zillow": get_zillow,
    "coauthor": get_coauthor,
    "test": get_test,
    "nuimages": get_nuimages,
}
