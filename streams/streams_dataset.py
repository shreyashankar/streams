from streams.create_domain_matrices import name_to_func
from streams.utils import create_probabilities

import avalanche
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import typing

supported_datasets = name_to_func.keys()


class STREAMSDataset(object):
    def __init__(
        self,
        name: str,
        T: int = None,
        inference_window: int = 0,
        seed: int = 42,
        force_download: bool = False,
        **kwargs,
    ) -> None:
        """Constructor for STREAMSDataset.

        Args:
            name (str): Name of the dataset. Must be in our supported datasets.
            T (int, optional): How many time steps in the stream. Defaults to
                length of dataset.
            inference_window (int, optional): Size of window after current time
                step that can be used for "testing." Defaults to 0.
            seed (int, optional): Random seed. Defaults to 42.
            force_download (bool, optional): Whether to forcibly redownload the
                dataset. Defaults to False.

        Raises:
            ValueError: If dataset is not in supported datasets.
        """
        if name not in supported_datasets:
            raise ValueError(f"Dataset {name} is not supported")
        self._name = name  # TODO(shreyashankar): make properties
        self._T = T
        self._inference_window = inference_window
        self._seed = seed
        logging.info(f"Creating dataset {name}")
        self.dataset, self.domain_matrices = name_to_func[self._name](
            force_download
        )

        # Create probabilities
        logging.info("Creating probabilities")
        self.sampling_probabilities, self.signals = create_probabilities(
            self.domain_matrices,
            T=len(self.dataset) if T is None else T,
            **kwargs,
        )

        # Create permutation
        self.permutation = np.array(
            [
                np.random.choice(len(prob), p=prob)
                for prob in self.sampling_probabilities
            ]
        )
        self.reset()

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

    @property
    def inference_window(self) -> int:
        return self._inference_window

    def reset(self) -> None:
        """Sets step back to 1."""
        self._step = 1

    def get_data(
        self,
        include_test=False,
    ) -> typing.Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Returns data up until current step.

        Args:
            include_test (bool, optional): Whether to include inference data.
                Defaults to False.

        Returns:
            typing.Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
                "Train" and test datasets.
        """
        train_dataset = self.get(list(range(self._step)))
        if not include_test:
            return train_dataset, None

        # Include test data
        test_dataset = self.get(
            list(range(self._step, self._step + self._inference_window)),
            future_ok=True,
        )
        return train_dataset, test_dataset

    def get_loaders(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        include_test: bool = False,
    ) -> typing.Tuple[
        torch.utils.data.DataLoader, torch.utils.data.DataLoader
    ]:
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

        test_dl = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size
        )
        return train_dl, test_dl

    def get_benchmark(
        self,
    ) -> avalanche.benchmarks.scenarios.new_instances.ni_scenario.NIScenario:
        """Gets data until current step as avalanche benchmark."""
        _, test_dataset = self.get_data(include_test=True)
        exp_assignment = [list(range(self._step))]
        benchmark = avalanche.benchmarks.generators.ni_benchmark(
            train_dataset=self.dataset,
            test_dataset=test_dataset,
            n_experiences=1,
            seed=self.seed,
            fixed_exp_assignment=exp_assignment,
        )
        return benchmark

    def advance(self, step_size: int = 1) -> None:
        """Advances the dataset by step_size.

        Args:
            step_size (int, optional): Defaults to 1.

        Raises:
            ValueError: If step_size is negative or outside dataset length.
        """
        if step_size < 1:
            raise ValueError("step_size must be at least 1")
        end = len(self.permutation) - self._inference_window
        if self._step + step_size > end:
            interval = end - self._step
            if interval <= 0:
                raise (ValueError("No more data to sample"))
            self._step += interval
            logging.info(f"Advanced {interval} steps.")
            return

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
        self, indices: typing.List[int], future_ok: bool = False
    ) -> torch.utils.data.Dataset:
        """Gets data for given indices. Reads into permutation and then
        accesses the data.

        Args:
            indices (typing.List[int]): Indices to get data for.
            future_ok (bool, optional): Whether it's ok to get data after
                current step. Defaults to False.

        Raises:
            ValueError: If indices are beyond current step.

        Returns:
            torch.utils.data.Dataset
        """
        if not future_ok:
            if isinstance(indices, int):
                if indices >= self._step:
                    raise ValueError(
                        f"Cannot sample index {indices} from future when"
                        + f" the current step is {self._step}."
                    )
            else:
                for index in indices:
                    if index >= self._step:
                        raise ValueError(
                            f"Cannot sample index {index} from future when"
                            + f" the current step is {self._step}."
                        )

        permutation_indices = self.permutation[indices]
        if isinstance(permutation_indices, np.int64):
            return self.dataset[permutation_indices]

        return torch.utils.data.Subset(self.dataset, permutation_indices)

    def __len__(self) -> int:
        """Gets length of dataset."""
        return len(self.permutation)

    def __getitem__(
        self, indices: typing.List[int]
    ) -> torch.utils.data.Dataset:
        """Accesses data for given indices. Wraps get."""
        return self.get(indices, future_ok=False)
