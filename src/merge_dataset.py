from dataset import HappinessDataset, MarriageDataset, IQDataset, SchoolingDataset, MeatDataset
import pandas as pd
import numpy as np
import pycountry
from ctgan import CTGAN
from ctgan import load_demo

def merge(DATA, key):
    for k, v in DATA.items():
        if k != key:
            v = v.rename(columns={'Country/region': 'country'})
            v = v.rename(columns={'Country': 'country'})
            
            DATA[key] = DATA[key].merge(v, on='country', how='left')
    return DATA[key]

def merge_datasets(happiness=True, marriage=True, iq=True,
                   meat_data=None, 
                   schooling_data=None, 
                   weather_data=None,
                   alchohol_data=None):
    DATA = {}
    key = 'HAPPINESS'
    if happiness:
        DATA['HAPPINESS'] = HappinessDataset.from_kaggle()['2023']
    if marriage:
        DATA['MARRIAGE'] = MarriageDataset.from_kaggle()['DATA1']
        # Drop data source year column
        DATA['MARRIAGE'] = DATA['MARRIAGE'].drop(columns=['Data Source Year'])
    if iq:
        DATA['IQ'] = IQDataset.from_kaggle()['DATA']
    if schooling_data is not None:
        DATA['SCHOOLING'] = schooling_data
        
    if meat_data is not None:
        DATA['MEAT'] = pd.read_csv('data/meat_data_2023.csv')
    if alchohol_data is not None:
        DATA['ALCOHOL'] = alchohol_data
    if weather_data is not None:
        DATA['WEATHER'] = weather_data
    
    final = merge(DATA, key)
    # Save the data
    final.to_csv('final.csv')
    return final
    
class Gen_Synth_Data:
    _OPTIONS = ['noise', 
                'ctgan',
                'copula',
                'vae']
    
    def __init__(self, data=None):
        self.data = data
        # fill nan values with average of the column
        self.numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.data = self.data[self.numeric_cols]
        self.data = self.data.fillna(self.data.mean())
    
        
    def generate(self, option='noise'):
        if option not in self._OPTIONS:
            raise ValueError(f'Invalid option. Choose from {self._OPTIONS}')
        print(f'Generating synthetic data using {option}...')
        method = {
            'noise': self.gen_by_noise,
            'ctgan': self.gen_by_ctgan,
            # 'copula': self.gen_by_copula,
            # 'vae': self.gen_by_vae
        }.get(option)
        return method()
    
    def gen_by_noise(self, noise_level=0.05, num_copies=5):
        """
        Generate synthetic data by adding Gaussian noise to the numeric columns.
        I
        """
        assert self.data is not None, 'Data is not loaded'

        synthetic_dfs = []
        for i in range(num_copies):

            noisy_df = self.data.copy()
            
            for col in self.numeric_cols:
                noise = np.random.normal(0, noise_level * self.data[col].std(), size=self.data[col].shape)
                noisy_df[col] += noise

            synthetic_dfs.append(noisy_df)
        
        return pd.concat([self.data] + synthetic_dfs, axis=0).reset_index(drop=True)
    
    def gen_by_ctgan(self, num_samples=1000):
        """
        Generate synthetic data using CTGAN.
        """
        assert self.data is not None, 'Data is not loaded'
        
        print(f'Generating synthetic data using CTGAN with {num_samples} samples...')
        data = self.data.select_dtypes(include=['number'])
        
        ctgan = CTGAN(epochs=10)
        ctgan.fit(data)

        return ctgan.sample(num_samples)
    
    
