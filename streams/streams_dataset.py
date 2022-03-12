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
        inference_window: int = 100,
        seed: int = 42,
        force_download: bool = False,
        **kwargs,
    ) -> None:
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
        self.permutation = [
            np.random.choice(len(prob), p=prob)
            for prob in self.sampling_probabilities
        ]
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
        self._step = 1

    def get_data(
        self,
        include_test=False,
    ) -> typing.Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        train_dataset = torch.utils.data.Subset(
            self.dataset, list(range(self._step))
        )
        if not include_test:
            return train_dataset, None

        # Include test data
        test_dataset = torch.utils.data.Subset(
            self.dataset,
            list(range(self._step, self._step + self._inference_window)),
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
        # TODO(shreyashankar): think about edge cases here
        end = len(self.permutation) - self._inference_window
        if self._step + step_size > end:
            interval = end - self._step
            if interval <= 0:
                raise (ValueError("No more data to sample"))
            self._step += interval
            return

        self._step += step_size

    def visualize(
        self, domain_type_index: int = 0, domain_value_indices: list = None
    ) -> None:
        if domain_value_indices is None:
            domain_value_indices = list(
                range(self.domain_matrices[domain_type_index].shape[1])
            )
        for i in domain_value_indices:
            x = list(range(self.T))
            y = [self.signals[t][domain_type_index][i] for t in x]
            plt.plot(x, y, label=f"Value {i}")

        # plt.ylim([-1, 10])
        plt.title(f"Domain {domain_type_index}")
        plt.rcParams["figure.figsize"] = (10, 4)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()

    def __len__(self) -> int:
        return len(self.permutation)

    def __getitem__(self, indices) -> torch.utils.data.Dataset:
        permutation_indices = self.permutation[indices]
        if isinstance(permutation_indices, int):
            return self.dataset[permutation_indices]

        return torch.utils.data.Subset(self.dataset, permutation_indices)
