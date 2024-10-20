import kagglehub
import pandas as pd
from typing import Dict, Type, List
import re
import os
import logging

LOGGER = logging.getLogger(__name__)


class HappinessDataset:
    _KAGGLE_DATASET_PATH: str = (
        "sazidthe1/global-happiness-scores-and-factors"
    )
    _FILE_YEAR_PATTERN = re.compile(r"WHR_(\d{4}).csv")

    def __init__(
        self, year_to_happiness_time_series: Dict[str, pd.DataFrame]
    ) -> None:
        self._happiness_time_series = year_to_happiness_time_series
        self._years = sorted(
            list(year_to_happiness_time_series.keys())
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
        return self._happiness_time_series[year_str]

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

        data = cls(year_to_happiness_time_series=year_to_df)
        LOGGER.info(f"Loaded data for years: {data.get_years()}")

        return data
