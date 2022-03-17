from streams.create_domain_matrices import name_to_func
from streams.utils import create_logits, softmax

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
        seed: int = 42,
        force_download: bool = False,
        n_t: typing.List = [],
        **kwargs,
    ) -> None:
        if name not in supported_datasets:
            raise ValueError(f"Dataset {name} is not supported")

        self._name = name  # TODO(shreyashankar): make properties
        self._T = T
        self._seed = seed
        logging.info(f"Creating dataset {name}")
        self.dataset, self.domain_matrices, self.time_periods = name_to_func[self._name](
            force_download
        )

        self.time_periods = None

        # Create probabilities
        logging.info("Creating logits")
        self.sampling_logits, self.signals = create_logits(
            self.domain_matrices,
            T=len(self.dataset) if T is None else T,
            **kwargs,
        )
        self.num_examples = len(self.sampling_logits[0])

        # Create samples
        self.sample_history = self._sample_without_replacement(n_t=n_t)

        self.reset()

    def _sample_without_replacement(self, n_t: typing.List=[]) -> typing.List[np.ndarray]:
        """For each 1..T time steps, draw a sample based on the corresponding
        probability distribution (without replacement). By default, any example
        can appear in any timestep. However, if time periods are specified (where
        lower-indexed time periods must appear earlier), then restrict when an 
        example can be sampled (e.g., all those appearing in time period 0 must be
        sample before any in time period 1).

        Args:
            n_t (optional): T-sized list with number of examples to be sampled
                for each timestep

        Returns:
            List of samples for timesteps 1 .. T.
        """
        if n_t == []:
            n_t = [ int(self.num_examples / self._T) ] * self._T

        time_periods = self.time_periods if (self.time_periods is not None) else np.zeros(self.num_examples)
        remaining = np.ones(self.num_examples)
        current_time_period = 0
        sample_history = []

        for t in range(self._T):
            goal = n_t[t]
            sample = []

            while len(sample) < goal:
                eligible = np.multiply(
                    remaining,
                    (time_periods <= current_time_period).astype(int)
                ).astype(bool)

                print(t, current_time_period, eligible.sum())

                if eligible.sum() > 0:
                    logits = self.sampling_logits[t].copy()
                    logits[~eligible] = -np.inf
                    probs = softmax(logits)

                    sample.extend(np.random.choice(
                        self.num_examples,
                        p=probs,
                        replace=False,
                        size=min(eligible.sum(), goal - len(sample))
                    ).tolist())

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
        self._step = 0

    def get_data(
        self,
        include_test=False,
    ) -> typing.Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        train_dataset = torch.utils.data.Subset(
            self.dataset,
            self.sample_history[self._step]
        )
        if not include_test:
            return train_dataset, None

        # Include test data
        test_dataset = torch.utils.data.Subset(
            self.dataset,
            self.sample_history[self._step + 1]
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
        if self._step + 1 == self._T - 1: # last timestep can't be used for training
            raise IOError("reached end of data stream")

        self._step += 1

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

    def __getitem__(self, t) -> torch.utils.data.Dataset:
        # examples in the same timestep do not have a particular order
        return torch.utils.data.Subset(self.dataset, self.sample_history[t])
