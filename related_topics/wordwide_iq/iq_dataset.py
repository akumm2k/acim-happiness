import kagglehub
import pandas as pd
from typing import Dict, Type, List
import re
import os
import logging

LOGGER = logging.getLogger(__name__)


class IQDataset:
    _KAGGLE_DATASET_PATH: str = (
        "abhijitdahatonde/worldwide-average-iq-levels"
    )

    def __init__(
        self, iq_data: Dict[str, pd.DataFrame]
    ) -> None:
        self._iq_data = iq_data
        
    def get_years(self) -> List[str]:
        return self._years

    def __getitem__(self, key) -> pd.DataFrame:
        if key == 'DATA':
            return self._iq_data

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
        year_to_df = {}
        
        assert len(csv_names) > 0, (
            "No CSV files found in the downloaded dataset."
        )
        assert len(csv_names) == 1, (
            "Expected 1 CSV file in the downloaded dataset. "
            f"Got {len(csv_names)}."
        )
        
        data = pd.read_csv(f'{path}/{csv_names[0]}')
        return data
