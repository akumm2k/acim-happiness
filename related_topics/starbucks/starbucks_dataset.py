import kagglehub
import pandas as pd
from typing import Dict, Type, List
import re
import os
import logging

LOGGER = logging.getLogger(__name__)

class StarbucksDataset:
    _KAGGLE_DATASET_PATH_1: str = (
        "kukuroo3/starbucks-locations-worldwide-2021-version"
    )
    _KAGGLE_DATASET_PATH_2: str = (
        "starbucks/store-locations"
    )
    def __init__(
        self, startbucks_data: Dict[str, pd.DataFrame]
    ) -> None:
        self._startbucks_data = startbucks_data
        
    def get_years(self) -> List[str]:
        return self._years

    def __getitem__(self, year) -> pd.DataFrame:
        year_str = str(year)
        if re.fullmatch(r"\d{2}", year_str) is not None:
            year_str = f"20{year_str}"

        assert re.fullmatch(r"\d{4}", year_str) is not None, (
            "Year must be a 4-digit number or a 2-digit number that "
            "can be converted to a 4-digit number by prepending '20'."
            f" Got {year_str}."
        )
        assert year_str in self._startbucks_data.keys(), (
            f"Year {year_str} not found in the dataset."
        )
        return self._startbucks_data[year_str]
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
        data = cls(startbucks_data=data)
        return data
