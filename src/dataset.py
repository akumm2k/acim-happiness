import kagglehub
import pandas as pd
from typing import Dict, Type, List
import re
import os
import logging

LOGGER = logging.getLogger(__name__)

class BaseDataset:
    _KAGGLE_DATASET_PATH: str = None
    _FILE_YEAR_PATTERN = re.compile(r"WHR_(\d{4}).csv")

    _IS_TIME_SERIES = False

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data

        if self._IS_TIME_SERIES:
            self._years = sorted(
                list(data.keys())
            )

    def get_years(self) -> List[str]:
        if not self._IS_TIME_SERIES:
            raise ValueError("This dataset is not a time series dataset.")
        return self._years

    def __getitem__(self, year: int) -> pd.DataFrame:
        if not self._IS_TIME_SERIES:
            raise ValueError("This dataset is not a time series dataset.")
        year_str = str(year)
        if re.fullmatch(r"\d{2}", year_str) is not None:
            year_str = f"20{year_str}"

        assert re.fullmatch(r"\d{4}", year_str) is not None, (
            "Year must be a 4-digit number or a 2-digit number that "
            "can be converted to a 4-digit number by prepending '20'."
            f" Got {year_str}."
        )
        return self._data[year_str]

class HappinessDataset(BaseDataset):
    _KAGGLE_DATASET_PATH: str = (
        "sazidthe1/global-happiness-scores-and-factors"
    )
    _FILE_YEAR_PATTERN = re.compile(r"WHR_(\d{4}).csv")
    _IS_TIME_SERIES = True

    def get_years(self) -> List[str]:
        return self._years

    @classmethod
    def from_kaggle(
        cls: Type["HappinessDataset"],
    ) -> "HappinessDataset":
        path = kagglehub.dataset_download(
            HappinessDataset._KAGGLE_DATASET_PATH
        )
        LOGGER.debug(f"Downloaded dataset to: {path}")

        csv_names = os.listdir(path)
        LOGGER.debug(f"Found CSV files: {csv_names}")

        year_to_df = {
            re.search(
                HappinessDataset._FILE_YEAR_PATTERN, csv_name
            ).group(1): pd.read_csv(f"{path}/{csv_name}")
            for csv_name in csv_names
        }

        data = cls(data=year_to_df)
        LOGGER.info(f"Loaded data for years: {data.get_years()}")

        return data

class AlcoholDataset(BaseDataset):
    _KAGGLE_DATASET_PATH: str = (
        "thedevastator/oecd-alcohol-consumption-per-capita/versions/2"
    )
    _FILE_YEAR_PATTERN = re.compile(r"WHR_(\d{4}).csv")
    _YEARS = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    _IS_TIME_SERIES = True

    @classmethod
    def from_kaggle(
        cls: Type["AlcoholDataset"],
    ) -> "AlcoholDataset":
        path = kagglehub.dataset_download(
            AlcoholDataset._KAGGLE_DATASET_PATH
        )
        LOGGER.debug(f"Downloaded dataset to: {path}")

        csv_names = os.listdir(path)
        LOGGER.debug(f"Found CSV files: {csv_names}")
        year_to_df = {}
        for csv_name in csv_names:
            data = pd.read_csv(f'{path}/{csv_name}')

            # Drop the 'INDEX' column if it exists
            if 'index' in data.columns:
                data = data.drop(columns=['index'])

            for year in cls._YEARS:
                if int(year) in data["TIME"].values:
                    if year not in year_to_df:
                        year_to_df[year] = data[data['TIME'] == int(year)].reset_index(drop=True)
                    else:
                        year_to_df[year] = pd.concat(
                            [year_to_df[year], data[data['TIME'] == int(year)].reset_index(drop=True)],
                            ignore_index=True
                        )

        data = cls(data=year_to_df)
        LOGGER.info(f"Loaded data for years: {data.get_years()}")

        return data

class MarriageDataset(BaseDataset):
    _KAGGLE_DATASET_PATH: str = (
        "johnny1994/divorce-rates-data-should-you-get-married"
    )

    def __getitem__(self, key: str) -> pd.DataFrame:
        assert key in ['DATA1', 'DATA2'], (
            f"Key must be one of ['DATA1', 'DATA2']. Got {key}."
        )
        return self._data[key]

    @classmethod
    def from_kaggle(
        cls: Type["MarriageDataset"],
    ) -> "MarriageDataset":
        path = kagglehub.dataset_download(
            MarriageDataset._KAGGLE_DATASET_PATH
        )
        LOGGER.debug(f"Downloaded dataset to: {path}")

        csv_names = os.listdir(path)
        LOGGER.debug(f"Found CSV files: {csv_names}")

        # Read specific sheets
        sheet1 = pd.read_excel(f'{path}/{csv_names[0]}', sheet_name='Divorce statistics by country_r')
        sheet2 = pd.read_excel(f'{path}/{csv_names[0]}', sheet_name='Estimates of annual divorces by')

        sheets = {
            'DATA1': sheet1,
            'DATA2': sheet2
        }

        data = cls(sheets)
        return data

class CoffeeDataset(BaseDataset):
    _KAGGLE_DATASET_PATH: str = (
        "michals22/coffee-dataset"
    )
    _FILE_YEAR_PATTERN = re.compile(r"WHR_(\d{4}).csv")
    _YEARS = ['2015', '2016', '2017', '2018', '2019']
    _IS_TIME_SERIES = True

    @classmethod
    def from_kaggle(
        cls: Type["CoffeeDataset"],
    ) -> "CoffeeDataset":
        path = kagglehub.dataset_download(
            CoffeeDataset._KAGGLE_DATASET_PATH
        )
        LOGGER.debug(f"Downloaded dataset to: {path}")

        csv_names = os.listdir(path)
        LOGGER.debug(f"Found CSV files: {csv_names}")
        year_to_df = {}

        csv_name = f"{path}/Coffee_domestic_consumption.csv"
        LOGGER.debug(f"Reading CSV file: {csv_name}")

        data = pd.read_csv(csv_name)

        # Drop the 'INDEX' column if it exists
        if 'index' in data.columns:
            data = data.drop(columns=['index'])

        data.columns = [
            re.sub(r"(\d{4})/\d{2}", r"\1", col) if re.match(r"(\d{4})/\d{2}", col) else col
            for col in data.columns
        ]

        for year in cls._YEARS:
            if year in data.columns:
                year_to_df[year] = (
                    data[["Country", year, 'Coffee type']]
                    .rename(columns={
                        year: 'Coffee_Consumption', 'Coffee type': 'Coffee_Type'
                    })
                )

        data = cls(data=year_to_df)
        LOGGER.info(f"Loaded data for years: {data.get_years()}")

        return data

class MeatDataset(BaseDataset):
    _KAGGLE_DATASET_PATH: str = (
        "ulrikthygepedersen/meat-consumption"
    )
    _FILE_YEAR_PATTERN = re.compile(r"WHR_(\d{4}).csv")
    _YEARS = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    _IS_TIME_SERIES = True

    @classmethod
    def from_kaggle(
        cls: Type["MeatDataset"],
    ) -> "MeatDataset":
        path = kagglehub.dataset_download(
            MeatDataset._KAGGLE_DATASET_PATH
        )
        LOGGER.debug(f"Downloaded dataset to: {path}")

        csv_names = os.listdir(path)
        LOGGER.debug(f"Found CSV files: {csv_names}")
        year_to_df = {}
        for csv_name in csv_names:
            data = pd.read_csv(f'{path}/{csv_name}')

            # Drop the 'INDEX' column if it exists
            if 'index' in data.columns:
                data = data.drop(columns=['index'])

            for year in cls._YEARS:
                if int(year) in data["time"].values:
                    if year not in year_to_df:
                        year_to_df[year] = data[data['time'] == int(year)].reset_index(drop=True)
                    else:
                        year_to_df[year] = pd.concat(
                            [year_to_df[year], data[data['time'] == int(year)].reset_index(drop=True)],
                            ignore_index=True
                        )

        data = cls(year_to_df)
        LOGGER.info(f"Loaded data for years: {data.get_years()}")

        return data

class SchoolingDataset(BaseDataset):
    _KAGGLE_DATASET_PATH: str = (
        "iamsouravbanerjee/years-of-schooling-worldwide"
    )

    def __getitem__(self, year) -> pd.DataFrame:
        year_str = str(year)
        assert re.fullmatch(r"\d{4}", year_str) is not None, (
            "Year must be a 4-digit number or a 2-digit number that "
            "can be converted to a 4-digit number by prepending '20'."
            f" Got {year_str}."
        )
        columns = ['ISO3', 'Country', 'Continent', 'Hemisphere',
                'Human Development Groups', 'UNDP Developing Regions',
                'HDI Rank (2021)', f'Expected Years of Schooling ({year_str})']

        return self._data[columns]

    @classmethod
    def from_kaggle(
        cls: Type["SchoolingDataset"],
    ) -> "SchoolingDataset":
        path = kagglehub.dataset_download(
            SchoolingDataset._KAGGLE_DATASET_PATH
        )
        LOGGER.debug(f"Downloaded dataset to: {path}")

        csv_names = os.listdir(path)
        LOGGER.debug(f"Found CSV files: {csv_names}")

        assert len(csv_names) > 0, (
            "No CSV files found in the downloaded dataset."
        )
        assert len(csv_names) == 1, (
            "Expected 1 CSV file in the downloaded dataset. "
            f"Got {len(csv_names)}."
        )

        data = pd.read_csv(f'{path}/{csv_names[0]}')
        data = cls(data)
        return data

class StarbucksDataset(BaseDataset):
    _KAGGLE_DATASET_PATH_1: str = (
        "kukuroo3/starbucks-locations-worldwide-2021-version"
    )
    _KAGGLE_DATASET_PATH_2: str = (
        "starbucks/store-locations"
    )

    def __getitem__(self, year) -> pd.DataFrame:
        year_str = str(year)
        if re.fullmatch(r"\d{2}", year_str) is not None:
            year_str = f"20{year_str}"

        assert re.fullmatch(r"\d{4}", year_str) is not None, (
            "Year must be a 4-digit number or a 2-digit number that "
            "can be converted to a 4-digit number by prepending '20'."
            f" Got {year_str}."
        )
        assert year_str in self._data.keys(), (
            f"Year {year_str} not found in the dataset."
        )
        return self._data[year_str]
    @classmethod
    def from_kaggle(
        cls: Type["StarbucksDataset"],
    ) -> "StarbucksDataset":
        path_1 = kagglehub.dataset_download(
            StarbucksDataset._KAGGLE_DATASET_PATH_1
        )
        path_2 = kagglehub.dataset_download(
            StarbucksDataset._KAGGLE_DATASET_PATH_2
        )
        LOGGER.debug(f"Downloaded dataset to: {path_1}")

        csv_names_1 = os.listdir(path_1)
        LOGGER.debug(f"Found CSV files 1: {csv_names_1}")

        csv_names_2 = os.listdir(path_2)
        LOGGER.debug(f"Found CSV files 2: {csv_names_2}")


        assert len(csv_names_2) > 0, (
            "No CSV files found in the downloaded dataset."
        )
        assert len(csv_names_2) == 1, (
            "Expected 1 CSV file in the downloaded dataset. "
            f"Got {len(csv_names_2)}."
        )
        assert len(csv_names_1) > 0, (
            "No CSV files found in the downloaded dataset."
        )
        assert len(csv_names_1) == 1, (
            "Expected 1 CSV file in the downloaded dataset. "
            f"Got {len(csv_names_1)}."
        )


        data_2021 = pd.read_csv(f'{path_1}/{csv_names_1[0]}')
        data_2017 = pd.read_csv(f'{path_2}/{csv_names_2[0]}')

        data = {'2021': data_2021, '2017': data_2017}
        data = cls(data)
        return data

class IQDataset(BaseDataset):
    _KAGGLE_DATASET_PATH: str = (
        "abhijitdahatonde/worldwide-average-iq-levels"
    )

    def __getitem__(self, key) -> pd.DataFrame:
        assert key == 'DATA', (
            f"Key must be 'DATA'. Got {key}."
        )
        return self._data

    @classmethod
    def from_kaggle(
        cls: Type["IQDataset"],
    ) -> "IQDataset":
        path = kagglehub.dataset_download(
            IQDataset._KAGGLE_DATASET_PATH
        )
        LOGGER.debug(f"Downloaded dataset to: {path}")

        csv_names = os.listdir(path)
        LOGGER.debug(f"Found CSV files: {csv_names}")

        assert len(csv_names) > 0, (
            "No CSV files found in the downloaded dataset."
        )
        assert len(csv_names) == 1, (
            "Expected 1 CSV file in the downloaded dataset. "
            f"Got {len(csv_names)}."
        )

        data = pd.read_csv(f'{path}/{csv_names[0]}')
        data = cls(data)
        return data

