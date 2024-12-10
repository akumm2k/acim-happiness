"""
Dear friends,
in merge dataset please look how it works. 
In short it merges our datasets into happiness dataset.
As I don't know how you used/processed your datasets, 
I created a solution where you just make dataset_name_data='to whatever data you used'.
E.g. merge_datasets(meat_data=my_meat_data) or merge_datasets(schooling_data=my_schooling_data) or merge_datasets(weather_data=my_weather_data) or merge_datasets(alchohol_data=my_alchohol_data)
"""

from sklearn.pipeline import Pipeline
import pandas as pd
from merge_dataset import merge_datasets
import pandas as pd
import numpy as np


def prepare_data():
    # Mock meat_dataset
    # mock_meat = pd.DataFrame({'country': ['USA', 'UK', 'China'], 'meat': [1, 2, 3]})
    
    # Please add your datas or send them to Mko and he will add them
    final_data = merge_datasets()

    return final_data

def stratified_split_by_region(data, test_size=0.15, val_size=0.15, random_state=42):
    """
    Splits the data into train, validation, and test sets by stratifying within regions.
    
    Parameters:
        data (pd.DataFrame): The dataset containing 'region' and 'country' columns.
        test_size (float): Proportion of countries to include in the test set within each region.
        val_size (float): Proportion of countries to include in the validation set within each region.
        random_state (int): Seed for reproducibility.
    
    Returns:
        pd.DataFrame: Training set.
        pd.DataFrame: Validation set.
        pd.DataFrame: Test set.
    """
    np.random.seed(random_state)
    
    train_list = []
    val_list = []
    test_list = []
    
    regions = data['region'].unique()
    
    for region in regions:
        region_data = data[data['region'] == region]
        countries = region_data['country'].unique()
        num_countries = len(countries)
        
        # Shuffle countries
        np.random.shuffle(countries)
        
        # Calculate the number of countries for each split
        test_split = int(np.floor(test_size * num_countries))
        val_split = int(np.floor(val_size * num_countries))
        
        # Ensure at least one country is selected if the region is small
        test_split = max(test_split, 1) if num_countries >= 3 else 0
        val_split = max(val_split, 1) if num_countries - test_split >= 3 else 0
        
        # Adjust splits if the number of countries is less than splits
        if test_split + val_split >= num_countries:
            test_split = max(num_countries - 2, 0)
            val_split = max(num_countries - test_split - 1, 0)
        
        train_split = num_countries - test_split - val_split
        
        # Split countries
        test_countries = countries[:test_split]
        val_countries = countries[test_split:test_split + val_split]
        train_countries = countries[test_split + val_split:]
        
        # Create DataFrames for each split
        train_region = region_data[region_data['country'].isin(train_countries)]
        val_region = region_data[region_data['country'].isin(val_countries)]
        test_region = region_data[region_data['country'].isin(test_countries)]
        
        # Append to lists
        train_list.append(train_region)
        val_list.append(val_region)
        test_list.append(test_region)
    
    # Concatenate all regions
    train = pd.concat(train_list, ignore_index=True)
    val = pd.concat(val_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)
    
    return train, val, test

# Example usage:
if __name__ == '__main__':
    # Assuming 'data' is your DataFrame with 'region' and 'country' columns
    data = prepare_data()  # Replace with your data loading function
    train, val, test = stratified_split_by_region(data, test_size=0.15, val_size=0.15, random_state=42)
    
    print(f"Total countries: {data['country'].nunique()}")
    print(f"Train countries: {train['country'].nunique()}")
    print(f"Validation countries: {val['country'].nunique()}")
    print(f"Test countries: {test['country'].nunique()}")