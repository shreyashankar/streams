"""Utility functions for creating streams."""

import io
import logging
import os
import random
import re
import typing
from datetime import datetime

import cvxpy as cp
import joblib
import nuimages
import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image


def aggregate_min(arrays: typing.List[np.ndarray], use_cvx: bool = True) -> np.ndarray:
    """Takes the minimum across all arrays.

    Args:
        arrays (typing.List[np.ndarray]): List of np arrays.
        use_cvx (bool, optional): Whether to use cvx. Defaults to True.

    Returns:
        np.ndarray: Min np array.
    """
    res = arrays[0]

    for i in range(1, len(arrays)):
        res = cp.minimum(res, arrays[i]) if use_cvx else np.minimum(res, arrays[i])

    return res


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x.

    Args:
        x (np.ndarray): Array to softmax.

    Returns:
        np.ndarray: Softmaxed array.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def create_logits(
    domain_matrices: typing.List[np.ndarray],
    T: int,
    gamma: float = 0.5,
    num_peaks: int = 5,
    start_max: int = 10,  # highest value signal can take to start with
    duration: int = 1,
    log_step: int = 10,
    starting_time_steps: typing.Dict[typing.Tuple[int, int], int] = {},
    seed: int = 0,
) -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
    """Creates logits and signals for T steps based on domain matrices.

    Args:
        domain_matrices (typing.List[np.ndarray]): List of domain matrices.
        T (int): _description_
        gamma (float, optional): Defaults to 0.5.
        num_peaks (int, optional): Defaults to 5.
        start_max (int, optional): Defaults to 10.
        starting_time_steps: A map of domain values to the timestep at which
            their signal value can be non-zero. For sufficiently large
            start_max, this precludes the chance of examples appearing early.
        seed (int, optional): Defaults to 0.

    Returns:
        typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
            List of logits vectors, list of signals
    """
    n = domain_matrices[0].shape[0]
    m = len(domain_matrices)
    logits = []
    signals = []

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)

    # signals in first period all set to same value for each domain type
    prev_s_vectors = [[start_max / 2] * mat.shape[1] for mat in domain_matrices]

    for (i, j) in starting_time_steps:
        if starting_time_steps[(i, j)] > 0:
            prev_s_vectors[i][j] = 0

    prev_z = aggregate_min(
        [mat @ s for mat, s in zip(domain_matrices, prev_s_vectors)],
        use_cvx=False,
    )
    prev_p = softmax(prev_z)

    signals.append(prev_s_vectors)
    logits.append(prev_z)

    peaks = np.random.choice(n, num_peaks).tolist()
    duration_counter = 0

    # Iterate
    for t in range(1, T, duration):
        c = np.ones(n)  # TODO(shreyashankar): change this when we do groups
        s_vectors = [cp.Variable(mat.shape[1]) for mat in domain_matrices]

        # z is concave in optimization variable s
        z = aggregate_min([mat @ s for mat, s in zip(domain_matrices, s_vectors)])

        # convex alternative to z (take mean instead of min over domain types)
        pseudo_z = (
            1
            / m
            * cp.sum([mat @ s for mat, s in zip(domain_matrices, s_vectors)], axis=1)
        )

        # instead of maximizing KL divergence with prev_p (not convex)
        # we find some entropic distribution p* with which
        # we can minimize KL divergence (convex)
        peaks.pop(0)
        peaks.append(np.random.randint(n))
        p_star = np.zeros(prev_p.shape)

        if len(peaks) == 1:
            p_star[peaks[0]] = 1
        else:
            p_star[peaks] = start_max
            p_star = softmax(p_star)

        obj = cp.Minimize(
            -1 * (p_star @ (np.log(c) + z - cp.log_sum_exp(pseudo_z + np.log(c))))
        )

        # prevent rapid changes from one timestep to another using L2 norm
        smoothness_constraints = [
            (
                1
                / (s_vectors[i].shape[0] ** 0.5)
                * cp.norm(s_vectors[i] - prev_s_vectors[i], 2)
            )
            <= gamma
            for i in range(m)
        ]

        # signal values should be non-negative
        nonnegativity_constraints = [s_vectors[i] >= 0 for i in range(m)] + [
            s_vectors[i] <= start_max for i in range(m)
        ]

        # only allow certain signal values past a certain timestep
        starting_time_step_constraints = [
            s_vectors[i][j] == 0
            for (i, j) in starting_time_steps.keys()
            if starting_time_steps[(i, j)] > t
        ]

        all_constraints = (
            smoothness_constraints
            + nonnegativity_constraints
            + starting_time_step_constraints
        )

        # Solve the problem
        prob = cp.Problem(obj, all_constraints)
        prob.solve()

        optimal_value = prob.value

        curr_z = aggregate_min(
            [mat @ s.value for mat, s in zip(domain_matrices, s_vectors)],
            use_cvx=False,
        )
        curr_p = softmax(curr_z)

        # signal value should persist for entirety of duration
        for d in range(duration):
            if len(signals) < T:
                signals.append([s.value for s in s_vectors])
                logits.append(curr_z)

            if (t + d) % log_step == 0:
                logging.info(f"Iteration {t + d}: {optimal_value}")

        # Set new prev vectors
        prev_s_vectors = [s_vec.value for s_vec in s_vectors]
        prev_z = curr_z
        prev_p = curr_p

    return logits, signals


class FullDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        transform: transforms.transforms = None,
    ):
        """Creates dataset for Avalanche.

        Args:
            dataset (WildsDataset, Torch dataset, whatever): Dataset to use.
            transform (transforms.transforms, optional): Defaults to None.
        """
        self.raw_dataset = dataset
        self.transform = transform
        self.targets = self.raw_dataset.y_array

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        x, y, metadata = self.raw_dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, metadata


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: typing.List[str],
        label_cols: typing.List[str],
        metadata_cols: typing.List[str] = [],
        transform: transforms.transforms = None,
    ):
        """Pytorch dataset for feature and target tensors.

        Args:
            df (pd.DataFrame): Dataframe to use.
            feature_cols (typing.List[str]): List of float-valued columns.
            label_cols (typing.List[str], optional): List of label columns.
            metadata_cols (typing.List[str], optional): List of
                gmetadata columns.
            transform (transforms.transforms, optional): Defaults to None.
        """
        self.transform = transform
        self.df = df
        self.feature_cols = feature_cols
        self.label_cols = label_cols if label_cols is not None else self.feature_cols
        self.targets = self.df[self.label_cols].values
        self.metadata_cols = metadata_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.iloc[idx][self.feature_cols].values
        y = self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)

        if self.metadata_cols:
            metadata = self.df.iloc[idx][self.metadata_cols].to_dict()
            return x, y, metadata

        return x, y


class RollingDataFrame(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: typing.List[str],
        group_col: str,
        label_cols: typing.List[str] = None,
        metadata_cols: typing.List[str] = [],
        transform: transforms.transforms = None,
    ):
        """Pytorch dataset for rolling windows of data.

        Args:
            df (pd.DataFrame): Dataframe to use.
            feature_cols (typing.List[str]): List of float-valued columns.
            group_col (str): Column to group by.
            label_cols (typing.List[str], optional): List of label columns.
            metadata_cols (typing.List[str], optional): List of
                gmetadata columns.
            transform (transforms.transforms, optional): Defaults to None.
        """
        self.transform = transform
        self.df = df
        self.feature_cols = feature_cols
        self.group_col = group_col
        self.label_cols = label_cols if label_cols is not None else self.feature_cols
        self.targets = self.df[self.label_cols].values
        self.metadata_cols = metadata_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        group_name = self.df.iloc[idx][self.group_col]
        subset = self.df.head(idx + 1)
        group_df = subset[subset[self.group_col] == group_name]
        x = group_df[self.feature_cols].values[:-1, :]
        y = group_df[self.label_cols].values[-1]

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        if self.transform is not None:
            x = self.transform(x)

        if self.metadata_cols:
            metadata = self.df.iloc[idx][self.metadata_cols].to_dict()
            return x, y, metadata

        return x, y


class NuImagesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        nuim: nuimages.nuimages.NuImages,
        labels: typing.List[int],
        metadata: dict,
        prefix: str,
        transform=None,
    ):
        """PyTorch dataset for NuImages.

        Args:
            nuim (nuimages.nuimages.NuImages): nuimages dataset.
            labels (typing.List[int]): List of labels.
            metadata (dict): Dictionary of modality, location, vehicle lists.
            prefix (str): Prefix for image filenames.
            transform: Torch transform. Defaults to None.
        """
        self.nuim = nuim
        self.metadata = metadata
        self.transform = transform
        self.targets = labels
        self.prefix = prefix

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        ann = self.nuim.object_ann[idx]
        category = self.nuim.get("category", ann["category_token"])
        sample_data = self.nuim.get("sample_data", ann["sample_data_token"])

        # Things to return
        filename = sample_data["filename"]
        bbox = ann["bbox"]
        category_name = category["name"]
        metadata = {key: self.metadata[key][idx] for key in self.metadata}

        full_path = os.path.join(self.prefix, filename)
        image = Image.open(full_path)
        x = transforms.Compose([transforms.PILToTensor()])(image)

        if self.transform is not None:
            x = self.transform(x)

        y = torch.tensor(self.targets[idx])

        return x, y, bbox, category_name, metadata


# UTILITY FUNCTIONS FOR COAUTHOR


def apply_ops(doc: str, mask: str, ops: list, source: str) -> typing.Tuple[str, str]:
    """Applies quilljs operations on a string. Taken
    from the CoAuthor website.

    Args:
        doc (str): Text so far (input document).
        mask (str): Character mask describing inputs.
        ops (list): List of JSON from quilljs.
        source (str): API or user.

    Returns:
        typing.Tuple[str, str]: Final string and mask after ops.
    """
    original_doc = doc
    original_mask = mask

    new_doc = ""
    new_mask = ""
    for i, op in enumerate(ops):

        # Handle retain operation
        if "retain" in op:
            num_char = op["retain"]

            retain_doc = original_doc[:num_char]
            retain_mask = original_mask[:num_char]

            original_doc = original_doc[num_char:]
            original_mask = original_mask[num_char:]

            new_doc = new_doc + retain_doc
            new_mask = new_mask + retain_mask

        # Handle insert operation
        elif "insert" in op:
            insert_doc = op["insert"]

            insert_mask = "U" * len(insert_doc)  # User
            if source == "api":
                insert_mask = "A" * len(insert_doc)  # API

            if isinstance(insert_doc, dict):
                if "image" in insert_doc:
                    logging.debug("Skipping invalid object insertion (image)")
                else:
                    logging.debug("Ignore invalid insertions:", op)
                    # Ignore other invalid insertions
                    # Debug if necessary
                    pass
            else:
                new_doc = new_doc + insert_doc
                new_mask = new_mask + insert_mask

        # Handle delete operation
        elif "delete" in op:
            num_char = op["delete"]

            if original_doc:
                original_doc = original_doc[num_char:]
                original_mask = original_mask[num_char:]
            else:
                new_doc = new_doc[:-num_char]
                new_mask = new_mask[:-num_char]

        else:
            # Ignore other operations
            # Debug if necessary
            logging.debug("Ignore other operations:", op)
            pass

    final_doc = new_doc + original_doc
    final_mask = new_mask + original_mask
    return final_doc, final_mask


def get_completion(text: str, mask: str) -> str:
    """Removes prompt from text.

    Args:
        text (str): Text generated so far.
        mask (str): Mask describing prompt or completion.

    Returns:
        str: text without prompt.
    """
    if "P" not in mask:
        return text
    end_index = mask.rindex("P") + 1
    return text[end_index:]


def get_prompts_and_completions(
    events: list,
    session_id: str,
    drop_keyword: str = "DROP_KEYWORD",
) -> pd.DataFrame:
    """For a session, returns a dataframe of prompts and completions.
    Every row in the dataframe represents a snapshot of the session
    taken every stride steps.

    Args:
        events (list): List of quilljs events for a session.
        session_id (str): Session ID.
        drop_keyword (str, optional): Keyword that represents drop. Defaults to
            "DROP_KEYWORD".

    Returns:
        pd.DataFrame: Prompts and completions for a session.
    """
    prompt = events[0]["currentDoc"].strip()

    text = prompt
    mask = "P" * len(prompt)  # Prompt

    texts = []
    timestamps = []
    for event in events:
        if "ops" not in event["textDelta"]:
            continue
        ops = event["textDelta"]["ops"]
        source = event["eventSource"]
        text, mask = apply_ops(text, mask, ops, source)

        # If the last char of text is a space, add it to texts
        if len(text) > 0 and text[-1] == " ":
            current_completion = get_completion(text, mask).strip()
            if current_completion != "" and re.search("[a-zA-Z]", current_completion):
                texts.append(current_completion)
                timestamps.append(event["eventTimestamp"])

    final_completion = get_completion(text, mask)
    df = pd.DataFrame(
        {
            "session_id": session_id.split(".")[0],
            "timestamp": timestamps,
            "prompt": prompt,
            "current": texts,
            "final": final_completion,
        }
    )
    # Filter duplicates
    df = df.drop_duplicates(subset=["current"], ignore_index=True)

    # Add metadata
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", origin="unix")
    df["next"] = df["current"].shift(-1)

    def subtract(row):
        if not isinstance(row["next"], str):
            return drop_keyword
        if row["current"].strip() not in row["next"]:
            return drop_keyword
        return row["next"].replace(row["current"].strip(), "")

    df["next"] = df.apply(subtract, axis=1)

    return df


def adult_filter(data: pd.DataFrame) -> pd.DataFrame:
    """Mimic the filters in place for Adult data.
    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    """
    df = data
    df = df[df["AGEP"] > 16]
    df = df[df["PINCP"] > 100]
    df = df[df["WKHP"] > 0]
    df = df[df["PWGTP"] >= 1]
    return df


def read_s3_config(name: str) -> dict:
    """Reads a config file from S3.

    Args:
        name (str): Name of config file.

    Returns:
        dict: Config file.
    """

    url = f"https://ocl-benchmarks.s3.us-west-1.amazonaws.com/configs/{name}.joblib"
    response = requests.get(url)
    data = io.BytesIO(response.content)
    data.seek(0)
    return joblib.load(data)
