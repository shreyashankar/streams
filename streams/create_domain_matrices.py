from dotenv import load_dotenv
from pathlib import Path
from torchvision import datasets
from streams.utils import FullDataset
from wilds import get_dataset

import numpy as np
import os
import pandas as pd
import subprocess
import torch
import torchvision.transforms as transforms
import typing

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


# download_path = os.path.join(HOME, DOWNLOAD_PREFIX, "iwildcam")
# command =
# f"kaggle competitions download -c iwildcam-2019-fgvc6 -p {download_path}"
# command += " --force" if force_download else ""

# subprocess.run("kaggle datasets files iwildcam-2019-fgvc6", shell=True)

# process = subprocess.run(command, shell=True)
# print("waiting")


name_to_func = {
    "mnist": get_mnist,
    "iwildcam": get_iwildcam,
    "civilcomments": get_civil_comments,
    "poverty": get_poverty,
}
