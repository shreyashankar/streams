import logging
import os
import typing

import avalanche
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from streams.create_domain_matrices import name_to_func
from streams.utils import create_logits, softmax

supported_datasets = name_to_func.keys()


class STREAMSDataset(object):
    def __init__(
        self,
        name: str,
        T: int = None,
        inference_window: int = 1,
        seed: int = 42,
        use_time_ordering: bool = False,
        force_download: bool = False,
        n_t: typing.List[int] = [],
        **kwargs,
    ) -> None:
        """Constructor for STREAMSDataset.

        Args:
            name (str): Name of the dataset. Must be in our supported datasets.
            T (int, optional): How many time steps in the stream. Defaults to
                length of dataset.
            inference_window (int, optional): Number of timesteps after current
                time step that can be used for "testing." Defaults to 1.
            seed (int, optional): Random seed. Defaults to 42.
            use_time_ordering (bool, optional): Whether to use time ordering.
            force_download (bool, optional): Whether to forcibly redownload the
                dataset. Defaults to False.
            n_t (typing.List[int], optional): T-sized list with number of
                examples to be sampled for each timestep.

        Raises:
            ValueError: If dataset is not in supported datasets.
        """
        if name not in supported_datasets:
            raise ValueError(f"Dataset {name} is not supported")

        self._name = name  # TODO(shreyashankar): make properties
        self._inference_window = inference_window
        self._seed = seed

        logging.info(f"Creating dataset {name}")
        (
            self.dataset,
            self.domain_matrices,
            self.time_periods,
            self.time_ordering,
        ) = name_to_func[self._name](force_download)
        self._n = self.domain_matrices[0].shape[0]
        self._T = T if T else self._n

        if self._n < self._T:
            raise ValueError("More timesteps than examples in dataset")

        self.time_periods = None

        # Create probabilities
        if not use_time_ordering:
            if "sampling_logits" in kwargs:
                self.sampling_logits = kwargs["sampling_logits"]
                self.signals = kwargs["signals"]
            else:
                logging.info("Creating logits")
                self.sampling_logits, self.signals = create_logits(
                    self.domain_matrices,
                    T=len(self.dataset) if T is None else T,
                    **kwargs,
                )

        # Create samples
        if "sample_history" in kwargs:
            self.sample_history = kwargs["sample_history"]
        else:
            self.sample_history = (
                self._sample_without_replacement(n_t=n_t)
                if not use_time_ordering
                else self._time_order(n_t=n_t)
            )
        self.num_examples = sum(
            [len(x) for x in self.sample_history]
        )  # could be less than self._n

        self.reset()

    def get_config(self) -> dict:
        """Gets configuraton of the stream.

        Returns:
            dict: Config serialized.
        """
        ret = {
            "name": self._name,
            "T": self._T,
            "inference_window": self._inference_window,
            "seed": self._seed,
            "use_time_ordering": self.time_ordering is not None,
            "sampling_logits": self.sampling_logits,
            "signals": self.signals,
            "sample_history": self.sample_history,
        }

        return ret

    @staticmethod
    def from_config(config_path: str) -> "STREAMSDataset":
        """Loads dataset from config file.

        Args:
            config_path (str): Path to config file.

        Returns:
            STREAMSDataset: Dataset loaded from config file.
        """
        return STREAMSDataset(**joblib.load(config_path))

    def save_config(self, path: str) -> None:
        """Saves configuration of the stream.

        Args:
            path (str): Path to save configuration.
        """
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        with open(path, "wb") as f:
            joblib.dump(self.get_config(), f)

    def _time_order(self, n_t: typing.List[int] = []) -> typing.List[np.ndarray]:
        """If user specifies a strict time ordering, then create the stream in
        this fashion.

        Args:
            n_t (optional): T-sized list with number of examples to be sampled
                for each timestep

        Returns:
            List of samples for timesteps 1 .. T.
        """
        if self.time_ordering is None:
            raise ValueError("No time ordering for this dataset.")

        if n_t == []:
            n_t = [int(self._n / self._T)] * self._T
            n_t[0] += self._n - sum(n_t)

        sample_history = []
        seen = 0

        for t in range(self._T):
            goal = n_t[t]
            sample = self.time_ordering[seen : seen + goal]
            sample_history.append(sample)
            seen += goal

        return sample_history

    def _sample_without_replacement(
        self, n_t: typing.List[int] = []
    ) -> typing.List[np.ndarray]:
        """For each 1..T time steps, draw a sample based on the corresponding
        probability distribution (without replacement). By default, any example
        can appear in any timestep. However, if time periods are specified
        (where lower-indexed time periods must appear earlier), then restrict
        when an example can be sampled (e.g., all those appearing in time
        period 0 must be sample before any in time period 1).

        Args:
            n_t (optional): T-sized list with number of examples to be sampled
                for each timestep

        Returns:
            List of samples for timesteps 1 .. T.
        """
        if n_t == []:
            n_t = [int(self._n / self._T)] * self._T
            n_t[0] += self._n - sum(n_t)

        time_periods = (
            self.time_periods if (self.time_periods is not None) else np.zeros(self._n)
        )
        remaining = np.ones(self._n)
        current_time_period = 0
        sample_history = []

        for t in range(self._T):
            goal = n_t[t]
            sample = []

            while len(sample) < goal:
                eligible = np.multiply(
                    remaining,
                    (time_periods <= current_time_period).astype(int),
                ).astype(bool)

                if eligible.sum() > 0:
                    logits = self.sampling_logits[t].copy()
                    logits[~eligible] = -np.inf
                    probs = softmax(logits)

                    sample.extend(
                        np.random.choice(
                            self._n,
                            p=probs,
                            replace=False,
                            size=min(eligible.sum(), goal - len(sample)),
                        ).tolist()
                    )

                    remaining[sample] = 0

                if len(sample) < goal:
                    current_time_period += 1
                    if current_time_period > time_periods.max():
                        break

            sample_history.append(sample)

        return sample_history

    @property
    def step(self) -> int:
        return self._step

    @property
    def name(self) -> str:
        return self._name

    @property
    def T(self) -> int:
        return self._T

    @property
    def seed(self) -> int:
        return self._seed

    def reset(self) -> None:
        """Sets current time step back to 0."""
        self._step = 0

    def get_data(
        self,
        include_test=False,
    ) -> typing.Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Returns data in the current step.

        Args:
            include_test (bool, optional): Whether to include inference data.
                Defaults to False.

        Returns:
            typing.Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
                "Train" and test datasets.
        """
        train_dataset = self.get([self._step])
        if not include_test:
            return train_dataset, None

        # Include test data
        test_dataset = self.get(
            list(range(self._step + 1, self._step + 1 + self._inference_window)),
            future_ok=True,
        )
        return train_dataset, test_dataset

    def get_loaders(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        include_test: bool = False,
    ) -> typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Dataloader wrapper around get_data.

        Args:
            batch_size (int, optional): Defaults to 64.
            shuffle (bool, optional): Defaults to True.
            include_test (bool, optional): Whether to include inference data.
                Defaults to False.

        Returns:
            typing.Tuple[ torch.utils.data.DataLoader,
                torch.utils.data.DataLoader ]: Dataloaders wrapped around the
                datasets from get_data.
        """
        train_dataset, test_dataset = self.get_data(include_test=include_test)
        train_dl = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        if not include_test:
            return train_dl, None

        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        return train_dl, test_dl

    def get_benchmark(
        self,
    ) -> avalanche.benchmarks.scenarios.new_instances.ni_scenario.NIScenario:
        """Gets data until current step as avalanche benchmark."""
        _, test_dataset = self.get_data(include_test=True)
        exp_assignment = [list(range(self._step))]
        try:
            benchmark = avalanche.benchmarks.generators.ni_benchmark(
                train_dataset=self.dataset,
                test_dataset=test_dataset,
                n_experiences=1,
                seed=self.seed,
                fixed_exp_assignment=exp_assignment,
            )
            return benchmark
        except ValueError:
            raise TypeError("Only classification tasks are supported by Avalanche.")

    def advance(self, step_size: int = 1) -> None:
        """Advances the current timestep by step_size.

        Args:
            step_size (int, optional): Defaults to 1.

        Raises:
            ValueError: If step_size is negative or outside dataset length.
        """
        if step_size < 1:
            raise ValueError("step_size must be at least 1")

        if (
            self._step + step_size > self._T - 1 - self._inference_window
        ):  # last timestep can't be used for training
            raise ValueError("reached end of data stream")

        self._step += step_size
        logging.info(f"Advanced {step_size} steps.")

    def visualize(
        self, domain_type_index: int = 0, domain_value_indices: list = None
    ) -> None:
        """Returns plot of signals for domain index and values.

        Args:
            domain_type_index (int, optional):Defaults to 0.
            domain_value_indices (list, optional): Defaults to None.
        """
        if domain_value_indices is None:
            domain_value_indices = list(
                range(self.domain_matrices[domain_type_index].shape[1])
            )
        for i in domain_value_indices:
            x = list(range(self.T))
            y = [self.signals[t][domain_type_index][i] for t in x]
            plt.plot(x, y, label=f"Value {i}")

        plt.title(f"Domain {domain_type_index}")
        plt.rcParams["figure.figsize"] = (10, 4)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()

    def get(
        self, step_indices: typing.List[int], future_ok: bool = False
    ) -> torch.utils.data.Dataset:
        """Gets data for given indices. Reads into permutation and then
        accesses the data.

        Args:
            step_indices (typing.List[int]): Indices to get data for.
            future_ok (bool, optional): Whether it's ok to get data after
                current step. Defaults to False.

        Raises:
            ValueError: If indices are beyond current step.

        Returns:
            torch.utils.data.Dataset
        """
        if not future_ok:
            for index in step_indices:
                if index > self._step:
                    raise ValueError(
                        f"Cannot sample index {index} from future when"
                        + f" the current step is {self._step}."
                    )

        data_indices = [i for t in step_indices for i in self.sample_history[t]]
        return torch.utils.data.Subset(self.dataset, data_indices)

    def __len__(self) -> int:
        """Gets length of dataset."""
        return len(self.sample_history)

    def __getitem__(self, step_indices: typing.List[int]) -> torch.utils.data.Dataset:
        """Accesses data for given indices. Wraps get."""
        return self.get(step_indices, future_ok=False)
