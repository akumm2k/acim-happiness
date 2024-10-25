import kagglehub
import pandas as pd
from typing import Dict, Type, List
import re
import os
import logging

LOGGER = logging.getLogger(__name__)


class AlcoholDataset:
    _KAGGLE_DATASET_PATH: str = (
        "thedevastator/oecd-alcohol-consumption-per-capita/versions/2"
    )
    _FILE_YEAR_PATTERN = re.compile(r"WHR_(\d{4}).csv")
    _YEARS = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']

    def __init__(
        self, year_to_alchohol_time_series: Dict[str, pd.DataFrame]
    ) -> None:
        self._alchohol_time_series = year_to_alchohol_time_series
        self._years = sorted(
            list(year_to_alchohol_time_series.keys())
        )

    def get_years(self) -> List[str]:
        return self._years

    def __getitem__(self, year: int) -> pd.DataFrame:
        year_str = str(year)
        if re.fullmatch(r"\d{2}", year_str) is not None:
            year_str = f"20{year_str}"

        assert re.fullmatch(r"\d{4}", year_str) is not None, (
            "Year must be a 4-digit number or a 2-digit number that "
            "can be converted to a 4-digit number by prepending '20'."
            f" Got {year_str}."
        )
        return self._alchohol_time_series[year_str]

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

        data = cls(year_to_alchohol_time_series=year_to_df)
        LOGGER.info(f"Loaded data for years: {data.get_years()}")

        return data
