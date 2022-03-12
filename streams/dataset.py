from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dotenv import dotenv_values

import os
import pandas as pd
import psycopg2
import typing


class BaseDataset(ABC):
    supported_datasets = ["taxi_data", "intel_lab_data"]
    start_cols = {
        "taxi_data": "tpep_pickup_datetime",
        "intel_lab_data": "reading_timestamp",
    }
    end_cols = {
        "taxi_data": "tpep_dropoff_datetime",
        "intel_lab_data": "reading_timestamp",
    }

    def __init__(
        self,
        name: str,
        cutoff_date: typing.Union[str, datetime],
        cache_dir: str = None,
    ):
        if name not in self.supported_datasets:
            raise ValueError(
                f"Dataset {name} is not supported. Supported datasets: {self.supported_datasets}"
            )
        self.name = name
        self.cutoff_date = (
            cutoff_date
            if isinstance(cutoff_date, datetime)
            else datetime.strptime(cutoff_date, "%Y-%m-%d")
        )
        self.cache_dir = cache_dir
        self.cache_dir = cache_dir
        self.connectToDB()

    def connectToDB(self):
        config = dotenv_values(".env")
        self.conn = psycopg2.connect(
            f"host={config.get('HOSTNAME')} user={config.get('USERNAME')} port={config.get('PORT')} password={config.get('SECRET')}"
        )
        self.conn.set_isolation_level(0)
        self.cur = self.conn.cursor()

    @abstractmethod
    def load(
        self,
        start_date: typing.Union[str, datetime],
        end_date: typing.Union[str, datetime],
    ):
        pass

    @abstractmethod
    def loadRecent(self, delta: timedelta):
        pass


class PandasDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        cutoff_date: typing.Union[str, datetime],
        cache_dir: str = None,
    ):
        super().__init__(name, cutoff_date, cache_dir=cache_dir)

    def load(
        self,
        start_date: typing.Union[str, datetime],
        end_date: typing.Union[str, datetime],
    ) -> pd.DataFrame:
        """Method to load data for the dataset.

        Args:
            start_date (typing.Union[str, datetime]): Start date of the data (inclusive).
            end_date (typing.Union[str, datetime]): End date of the data (exclusive).

        Raises:
            ValueError: When end date is before start date.
            ValueError: When end date is after cutoff date.

        Returns:
            pd.DataFrame: Loaded data for the dataset.
        """
        start_date = (
            start_date
            if isinstance(start_date, datetime)
            else datetime.strptime(start_date, "%Y-%m-%d")
        )
        end_date = (
            end_date
            if isinstance(end_date, datetime)
            else datetime.strptime(end_date, "%Y-%m-%d")
        )
        # Check that end is after start
        if end_date < start_date:
            raise ValueError("End date must be after start date")

        # Check if cutoff date is in the range
        if self.cutoff_date <= end_date:
            raise ValueError("Cutoff date must be after end date")

        # Create pandas df from db
        query = f"SELECT * FROM {self.name} WHERE {self.start_cols[self.name]} >= '{start_date}' AND {self.end_cols[self.name]} < '{end_date}';"
        df = pd.read_sql_query(query, self.conn)
        return df

    def loadRecent(self, delta: timedelta) -> pd.DataFrame:
        """Method to load recent data for the dataset.

        Args:
            delta (timedelta): How far back in time to load data.

        Returns:
            pd.DataFrame: Loaded data for the dataset.
        """
        start_date = self.cutoff_date - delta
        query = f"SELECT * FROM {self.name} WHERE {self.start_cols[self.name]} >= '{start_date}' AND {self.end_cols[self.name]} < '{self.cutoff_date}';"
        df = pd.read_sql_query(query, self.conn)
        return df


class Dataset:
    def __init__(
        self,
        name: str,
        cutoff_date: typing.Union[str, datetime],
        cache_dir: str = None,
        backend: str = "pandas",
    ):
        if backend not in ["pandas"]:
            raise ValueError("Backend not supported")

        if backend == "pandas":
            self.dataset = PandasDataset(name, cutoff_date, cache_dir)

    def load(
        self,
        start_date: typing.Union[str, datetime],
        end_date: typing.Union[str, datetime],
    ) -> pd.DataFrame:
        """Method to load data for the dataset.

        Args:
            start_date (typing.Union[str, datetime]): Start date of the data (inclusive).
            end_date (typing.Union[str, datetime]): End date of the data (exclusive).

        Raises:
            ValueError: When end date is before start date.
            ValueError: When end date is after cutoff date.

        Returns:
            pd.DataFrame: Loaded data for the dataset.
        """

        return self.dataset.load(start_date, end_date)

    def loadRecent(self, delta: timedelta) -> pd.DataFrame:
        """Method to load recent data for the dataset.

        Args:
            delta (timedelta): How far back in time to load data.

        Returns:
            pd.DataFrame: Loaded data for the dataset.
        """

        return self.dataset.loadRecent(delta)


# TODO (shreyashankar): Add support for other datasets
# (TF, Pandas)
