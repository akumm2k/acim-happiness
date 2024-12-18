from xgboost import XGBRegressor
import pandas as pd
import os
import joblib
import pickle

class CustomDataPreprocessor:
    
    drop_columns = ['country', 'Continent_x', 'Continent_y', 'Year', 'year', 'Unnamed: 0', 'latitude', 'longitude', 'altitude', 'avg_temp_c']
    categorical_columns = ['region', 'Hemisphere']
    binary_categorical_columns = ['beef', 'pig', 'poultry', 'sheep'] # check is_na or not


    unclear_columns = ['rank', 'Percent', 'Actual', 'LOCATION', 'INDICATOR', 'SUBJECT',  'MEASURE', 'FREQUENCY', 'TIME', 'ISO3', 'Human Development Groups' ,'UNDP Developing Regions']

    a_lot_nas = ['beef', 'pig', 'poultry', 'sheep', 'LITRES/CAPITA']
    half_nans = ['avg_income', 'Marriage', 'Divorce', 'Actual', 'Percent']
    one_third_nans = ['rank', 'IQ', 'education_expenditure', 'avg_income', 'avg_temp']
    small_nas = ['HDI Rank (2021)', 'Expected Years of Schooling']
    
    def __init__(self, fill_method={'half_nans': 'median', 'one_third_nans': 'median', 'small_nas': 'mode', 'unclear_columns': 'drop'}):
        
        self.fill_method = fill_method or {}
        self.column_mapping = {} 
        self.all_columns = None
        self.fitted = False
        
    def fit(self, df):
        
        self.column_mapping = {}
        self.all_columns = None
        self.fitted = False
        
        if self.categorical_columns:
            for col in self.categorical_columns:
                if col in df:
                    self.column_mapping[col] = df[col].dropna().unique()
        
        self.all_columns = (
            set(self.categorical_columns or []) |
            set(self.binary_categorical_columns or []) |
            set(self.unclear_columns or []) |
            set(self.a_lot_nas or []) |
            set(self.half_nans or []) |
            set(self.one_third_nans or []) |
            set(self.small_nas or [])
        )
        
        self.all_columns -= set(self.drop_columns or [])
        self.all_columns = self.all_columns.intersection(df.columns)

        self.fitted = True

    def transform(self, df_init):
        if not self.fitted:
            raise RuntimeError("The transformer has not been fitted. Call 'fit' first.")

        df = df_init.copy()
        
        # Step 0: Retain only columns explicitly specified
        missing_columns = self.all_columns - set(df.columns)
        for col in missing_columns:
            df[col] = 0  # Add missing columns with default value 0
            
        df = df.loc[:, list(self.all_columns)]
        
        # Step 1: Drop unnecessary columns
        if self.drop_columns:
            df.drop(columns=self.drop_columns, inplace=True, errors="ignore")

        # Step 2: Handle categorical columns
        if self.categorical_columns:
            for col, unique_values in self.column_mapping.items():
                if col in df:
                    # Create dummy columns based on stored categories
                    for val in unique_values:
                        col_name = f"{col}_{val}"
                        df.loc[:, col_name] = (df[col] == val).astype(int)
                    # Drop the original categorical column
                    df.drop(columns=[col], inplace=True)

        # Step 3: Handle binary categorical columns (based on NaN presence)
        if self.binary_categorical_columns:
            for col in self.binary_categorical_columns:
                if col in df:
                    df.loc[:, col] = df[col].isna().astype(int)

        # Step 4: Drop columns with many NaN values
        if self.a_lot_nas:
            df.drop(columns=self.a_lot_nas, inplace=True, errors="ignore")

        columns_with_nans = {
            "half_nans": self.half_nans,
            "one_third_nans": self.one_third_nans,
            "small_nas": self.small_nas,
            "unclear_columns": self.unclear_columns,
        }
        for method, columns in columns_with_nans.items():
            if columns:
                for col in columns:
                    if col in df:
                        fill_strategy = self.fill_method.get(method)
                        if fill_strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].median())
                        elif fill_strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].mean())
                        elif fill_strategy == "mode":
                            mode_value = df[col].mode()
                            if not mode_value.empty:
                                df[col] = df[col].fillna(mode_value[0])
                        elif fill_strategy == "ffill":
                            df[col] = df[col].ffill()
                        elif fill_strategy == "bfill":
                            df[col] = df[col].bfill()
                        elif fill_strategy == "drop":
                            df.drop(columns=[col], inplace=True, errors="ignore")

        return df

    def fit_transform(self, df):
        """
        Combines fit and transform into a single method for convenience.
        """
        self.fit(df)
        return self.transform(df)

class CustomModel:
    model = XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.1, random_state=42)
    
    data_path: str = 'data/noise_synthetic_data.csv'
    target_column = 'happiness_score'
    model_path = 'data/model.pkl'
    
    DEBUG = False
    
    def __init__(self):
        self.preprocessor = CustomDataPreprocessor()
        self.train()
    
    def preprocess_fit_tranform(self, X):
        return self.preprocessor.fit_transform(X)
    
    def preprocess(self, X):
        return self.preprocessor.transform(X)
    
    def train(self, preprocess_required=True):
        
        # if os.path.exists(self.model_path):
            
        #     print('Model already exists. Skipping training...')
        #     print('Loading the model...')
            
        #     self.model = pickle.load(open(self.model_path, "rb"))
            
        #     return
        
        data = pd.read_csv(self.data_path)
        
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]
        
        if self.DEBUG:
            print('Preprocessing data...')
            
        if preprocess_required:
            X = self.preprocess_fit_tranform(X)
            
        if self.DEBUG: 
            print('Training the model...')
        
        self.model.fit(X, y)
        
        if self.DEBUG:
            print('Model trained successfully')
        
        # save the model
        import joblib
        joblib.dump(self.model, self.model_path)
    
    def predict(self, X, preprocess_required=True):
        
        if self.DEBUG:
            print("X before preprocess:", X)
        if preprocess_required:
            X = self.preprocess(X)
            
            if self.DEBUG:
                print("X after preprocess:", X) 
                     
        return self.model.predict(X)
    
    
model = CustomModel()

sample = pd.read_csv('data/noise_synthetic_data.csv').head(2)

print("Sample data:", sample)

print("predict : ", model.predict(sample))

