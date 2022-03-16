"""Utility functions for creating streams."""

from datetime import datetime

import cvxpy as cp
import logging
import numpy as np
import pandas as pd
import random
import torch
import torchvision.transforms as transforms
import typing


def aggregate_min(
    arrays: typing.List[np.ndarray], use_cvx: bool = True
) -> np.ndarray:
    """Takes the minimum across all arrays.

    Args:
        arrays (typing.List[np.ndarray]): List of np arrays.
        use_cvx (bool, optional): Whether to use cvx. Defaults to True.

    Returns:
        np.ndarray: Min np array.
    """
    res = arrays[0]

    for i in range(1, len(arrays)):
        res = (
            cp.minimum(res, arrays[i])
            if use_cvx
            else np.minimum(res, arrays[i])
        )

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


def create_probabilities(
    domain_matrices: typing.List[np.ndarray],
    T: int,
    gamma: float = 0.5,
    num_peaks: int = 5,
    start_max: int = 10,  # highest value signal can take to start with
    duration: int = 1,
    log_step: int = 10,
    seed: int = 0,
    periodicity_slack: float = 2,
    periodicity: typing.List[
        typing.Tuple[
            int, int
        ]  # change this param to how many cycles istead of period
    ] = [],  # tuple of domains and periods
) -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
    """Creates probability and signals for T steps based on domain matrices.

    Args:
        domain_matrices (typing.List[np.ndarray]): List of domain matrices.
        T (int): _description_
        gamma (float, optional): Defaults to 0.5.
        num_peaks (int, optional): Defaults to 5.
        start_max (int, optional): Defaults to 10.
        seed (int, optional): Defaults to 0.
        periodicity_slack (float, optional): Defaults to 2.
        periodicity (typing.List[ typing.Tuple[ int, int ], optional):
            TODO(shreyashankar: describe). Defaults to [].

    Returns:
        typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
            List of probability vectors, list of signals
    """
    n = domain_matrices[0].shape[0]
    m = len(domain_matrices)
    probabilities = []
    signals = []

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)

    prev_s_vectors = [
        [start_max / mat.shape[1]]
        * mat.shape[1]  # initialize to midpoint of range
        for mat in domain_matrices
    ]
    prev_z = aggregate_min(
        [mat @ s for mat, s in zip(domain_matrices, prev_s_vectors)],
        use_cvx=False,
    )
    prev_p = softmax(prev_z)

    signals.append(prev_s_vectors)
    probabilities.append(prev_p)

    peaks = np.random.choice(n, num_peaks).tolist()
    duration_counter = 0

    # Iterate
    for t in range(1, T + 1, duration):
        c = np.ones(n)  # TODO(shreyashankar): change this when we do groups
        s_vectors = [cp.Variable(mat.shape[1]) for mat in domain_matrices]

        # z is concave in optimization variable s
        z = aggregate_min(
            [mat @ s for mat, s in zip(domain_matrices, s_vectors)]
        )

        # convex alternative to z (take mean instead of min over domain types)
        pseudo_z = (
            1
            / m
            * cp.sum(
                [mat @ s for mat, s in zip(domain_matrices, s_vectors)], axis=1
            )
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
            -1
            * (p_star @ (np.log(c) + z - cp.log_sum_exp(pseudo_z + np.log(c))))
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

        nonnegativity_constraints = [s_vectors[i] >= 0 for i in range(m)]

        # value of jth value of domain i should be the same as (j-1)th value
        # 'period' timesteps ago
        periodic_constraints = []

        for i, period in periodicity:
            if t > period:
                periodic_constraints.append(
                    cp.norm(s_vectors[i] - np.roll(signals[-period][i], 1), 1)
                    <= periodicity_slack
                )

        all_constraints = (
            smoothness_constraints
            + nonnegativity_constraints
            + periodic_constraints
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
            if len(signals) <= T:
                signals.append([s.value for s in s_vectors])
                probabilities.append(curr_p)

            if (t + d) % log_step == 0:
                logging.info(f"Iteration {t + d}: {optimal_value}")

        # Set new prev vectors
        prev_s_vectors = [s_vec.value for s_vec in s_vectors]
        prev_z = curr_z
        prev_p = curr_p

    return probabilities, signals


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
        self.label_cols = (
            label_cols if label_cols is not None else self.feature_cols
        )
        self.targets = self.df[self.label_cols].values
        self.metadata_cols = metadata_cols

    def __len__(self):
        return len(self.features)

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
        self.label_cols = (
            label_cols if label_cols is not None else self.feature_cols
        )
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


### UTILITY FUNCTIONS FOR COAUTHOR ###


def apply_ops(
    doc: str, mask: str, ops: list, source: str
) -> typing.Tuple[str, str]:
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
                    logging.info("Skipping invalid object insertion (image)")
                else:
                    logging.info("Ignore invalid insertions:", op)
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
            logging.info("Ignore other operations:", op)
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
    end_index = mask.rindex("P")
    return text[end_index + 1 :]


def get_prompts_and_completions(
    events: list, session_id: str, stride: int = 10
) -> pd.DataFrame:
    """For a session, returns a dataframe of prompts and completions.
    Every row in the dataframe represents a snapshot of the session
    taken every stride steps.

    Args:
        events (list): List of quilljs events for a session.
        session_id (str): Session ID.
        stride (int, optional): Snapshot time. Defaults to 10.

    Returns:
        pd.DataFrame: Prompts and completions for a session.
    """
    prompt = events[0]["currentDoc"].strip()

    text = prompt
    mask = "P" * len(prompt)  # Prompt

    texts = []
    timestamps = []
    first = True
    for i, event in enumerate(events):
        if "ops" not in event["textDelta"]:
            continue
        ops = event["textDelta"]["ops"]
        source = event["eventSource"]
        text, mask = apply_ops(text, mask, ops, source)
        if first and list(ops[-1].keys())[0] in ["insert", "delete"]:
            texts.append(get_completion(text, mask))
            timestamps.append(event["eventTimestamp"])
            first = False

        elif i % stride == 0:
            texts.append(get_completion(text, mask))
            timestamps.append(event["eventTimestamp"])
            first = False

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
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", origin="unix")

    return df
