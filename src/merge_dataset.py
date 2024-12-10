from dataset import HappinessDataset, MarriageDataset, IQDataset, SchoolingDataset, MeatDataset
import pandas as pd

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
        DATA['MEAT'] = meat_data
    if alchohol_data is not None:
        DATA['ALCOHOL'] = alchohol_data
    if weather_data is not None:
        DATA['WEATHER'] = weather_data
    
    final = merge(DATA, key)
    # Save the data
    final.to_csv('final.csv')
    return final
    
def main():
    # mock datasets
    mock_schooling = pd.DataFrame({'country': ['USA', 'UK', 'China'], 'schooling': [1, 2, 3]})
    merge_datasets(schooling=True, meat=True)

if __name__ == '__main__':
    main()