from sklearn.pipeline import Pipeline
import pandas as pd
from merge_dataset import merge_datasets, Gen_Synth_Data
import pandas as pd
import numpy as np
import os

def prepare_data():
    # if data exists data/final_data.csv, then load it
    if os.path.exists('data/final_data.csv'):
        return pd.read_csv('data/final_data.csv')
    
    # Load the datasets
    alcohol_data = pd.read_csv('data/alcoholdataset.csv')
    schooling_data = pd.read_csv('data/schooling_data_long.csv')
    weather_data = pd.read_csv('data/weather_data.csv')

    # filter schooling data by only using 2021 data
    schooling_data = schooling_data[schooling_data['Year'] == 2021]
    
    meat_data = pd.read_csv('data/meat_data_2023.csv')
    # Drop city_name column from weather data
    weather_data = weather_data.drop(columns=['city_name'])
    # average the weather data by country
    weather_data = weather_data.groupby('country').mean().reset_index()

    # Please add your datas or send them to Mko and he will add them
    final_data = merge_datasets(schooling_data=schooling_data, alchohol_data=alcohol_data, meat_data=meat_data, weather_data=weather_data)
    #  Save the data
    final_data.to_csv('data/final_data.csv', index=False)
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
    
    # save it to csv
    train.to_csv('data/train_data.csv', index=False)
    val.to_csv('data/val_data.csv', index=False)
    test.to_csv('data/test_data.csv', index=False)
    return train, val, test

def generate_synth_data(data):
    """
    Generate synthetic data by adding Gaussian noise to the numerical columns.
    
    Parameters:
        data (pd.DataFrame): The dataset to generate synthetic data from.
    
    Returns:
        pd.DataFrame: The original dataset with synthetic data appended.
    """
    # drop the region, country columns
    data = data.drop(columns=['region', 'country'])
    gen = Gen_Synth_Data(data)
    
    # if noise data exists data/noise_synthetic_data.csv, then load it
    synth_noise = gen.generate('noise') if not os.path.exists('data/noise_synthetic_data.csv') else pd.read_csv('data/noise_synthetic_data.csv')
    # save it to csv if it doesn't exist
    if not os.path.exists('data/noise_synthetic_data.csv'):
        synth_noise.to_csv('data/noise_synthetic_data.csv', index=False)
        
    synth_ctgan = gen.generate('ctgan') if not os.path.exists('data/ctgan_synthetic_data.csv') else pd.read_csv('data/ctgan_synthetic_data.csv')
    # save it to csv if it doesn't exist
    # if not os.path.exists('data/ctgan_synthetic_data.csv'):
    #     synth_ctgan.to_csv('data/ctgan_synthetic_data.csv', index=False)
    
    return synth_noise, synth_ctgan

if __name__ == '__main__':
    data = prepare_data()  # Replace with your data loading function
    
    # Print numerical columns of data
    train, val, test = stratified_split_by_region(data, test_size=0.15, val_size=0.15, random_state=42)
    
    synth_noise, synth_ctgan = generate_synth_data(train)
    print(synth_ctgan)
    # save it to csv
    synth_noise.to_csv('data/noise_synthetic_data.csv', index=False)
    print(f"Total countries: {data['country'].nunique()}")
    print(f"Train countries: {train['country'].nunique()}")
    print(f"Validation countries: {val['country'].nunique()}")
    print(f"Test countries: {test['country'].nunique()}")