import kagglehub
import pandas as pd
from typing import Dict, Type, List
import re
import os
import logging

LOGGER = logging.getLogger(__name__)


class MarriageDataset:
    _KAGGLE_DATASET_PATH: str = (
        "johnny1994/divorce-rates-data-should-you-get-married"
    )

    def __init__(
        self, _marriage_data: Dict[str, pd.DataFrame]
    ) -> None:
        self._marriage_data = _marriage_data

    def __getitem__(self, key: str) -> pd.DataFrame:
        assert key in ['DATA1', 'DATA2'], (
            f"Key must be one of ['DATA1', 'DATA2']. Got {key}."
        )
        return self._marriage_data[key]

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
        
        data = cls(_marriage_data=sheets)
        return data
