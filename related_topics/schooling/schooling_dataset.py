import kagglehub
import pandas as pd
from typing import Dict, Type, List
import re
import os
import logging

LOGGER = logging.getLogger(__name__)


class SchoolingDataset:
    _KAGGLE_DATASET_PATH: str = (
        "iamsouravbanerjee/years-of-schooling-worldwide"
    )

    def __init__(
        self, schooling_data: Dict[str, pd.DataFrame]
    ) -> None:
        self._schooling_data = schooling_data
        
    def get_years(self) -> List[str]:
        return self._years

    def __getitem__(self, year) -> pd.DataFrame:
        print(year)
        year_str = str(year)
        assert re.fullmatch(r"\d{4}", year_str) is not None, (
            "Year must be a 4-digit number or a 2-digit number that "
            "can be converted to a 4-digit number by prepending '20'."
            f" Got {year_str}."
        )
        columns = ['ISO3', 'Country', 'Continent', 'Hemisphere',
                'Human Development Groups', 'UNDP Developing Regions',
                'HDI Rank (2021)', f'Expected Years of Schooling ({year_str})']
        
        return self._schooling_data[columns]

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
        data = cls(schooling_data=data)
        return data
