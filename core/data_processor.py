import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

class DataProcessor:
    
    def __init__(self, df):
        self.df = df.copy()

    def process_blood_pressure(self):
        if 'Blood Pressure' in self.df.columns:
            try:
                bp_split = self.df['Blood Pressure'].str.split('/', expand=True)
                self.df['Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
                self.df['Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')
                self.df = self.df.drop(columns=['Blood Pressure'])
                print('Đã xử lý cột huyết áp thành công.')
            except Exception as e:
                print(f'Lỗi khi xử lý cột huyết áp: {e}')
        return self.df
    
    def classify_variables(self):
        numerical_cols = []
        categorical_cols = []
        binary_cols = []
        known_binary = ['Diabetes', 'Family History', 'Smoking', 'Obesity',
                        'Alcohol Consumption', 'Previous Heart Problems', 'Medication Use', 'Heart Attack Risk']
        known_categorical = ['Sex', 'Diet', 'Country', 'Continent', 'Hemisphere']
        excluded_cols = ['Patient ID']
        if 'Heart Attack Risk' in self.df.columns:
            if 'Heart Attack Risk' not in binary_cols:
                binary_cols.append('Heart Attack Risk')
        for col in self.df.columns:
            if col in excluded_cols or col == 'Heart Attack Risk':
                continue
            if col in ['Systolic', 'Diastolic']:
                if col not in numerical_cols:
                    numerical_cols.append(col)
                continue
            if col in known_binary:
                if col not in binary_cols:
                    binary_cols.append(col)
                continue
            if col in known_categorical:
                if col not in categorical_cols:
                    categorical_cols.append(col)
                continue
            unique_vals = self.df[col].nunique()
            dtype = self.df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                if unique_vals == 2 and set(self.df[col].dropna().unique()).issubset({0, 1, True, False}):
                    if col not in binary_cols:
                        binary_cols.append(col)
                elif unique_vals < 15 and unique_vals > 2:
                    if col not in categorical_cols:
                        categorical_cols.append(col)
                else:
                    if col not in numerical_cols:
                        numerical_cols.append(col)
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                if unique_vals == 2: # Ví dụ: 'Male', 'Female'
                    if col not in binary_cols:
                        binary_cols.append(col)
                else:
                    if col not in categorical_cols:
                        categorical_cols.append(col)
        numerical_cols = sorted(list(set(numerical_cols)))
        categorical_cols = sorted(list(set(categorical_cols)))
        binary_cols = sorted(list(set(binary_cols)))

        return {
            'numerical': numerical_cols,
            'categorical': categorical_cols,
            'binary': binary_cols
        }
    
    def handle_missing_values(self, strategy='mean', numerical_cols=None, categorical_cols=None):
        df_processed = self.df.copy()
        if numerical_cols:
            if strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif strategy == 'median':
                imputer = SimpleImputer(strategy='median')
            else: 
                imputer = SimpleImputer(strategy='most_frequent')
            cols_to_impute = [col for col in numerical_cols if df_processed[col].isnull().any()]
            if cols_to_impute:
                df_processed[cols_to_impute] = imputer.fit_transform(df_processed[cols_to_impute])
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            cols_to_impute = [col for col in categorical_cols if df_processed[col].isnull().any()]
            if cols_to_impute:
                df_processed[cols_to_impute] = cat_imputer.fit_transform(df_processed[cols_to_impute])

        self.df = df_processed
        return self.df

    def handle_outliers_isolation_forest(self, numerical_cols=None, contamination=0.05):
        if numerical_cols is None:
            return self.df
        df_processed = self.df.copy()
        for col in numerical_cols:
            if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col]):
                data_for_model = df_processed[col].dropna()
                if len(data_for_model) > 10:
                    iso_forest = IsolationForest(contamination=contamination, random_state=42, warm_start=True)
                    outliers = iso_forest.fit_predict(data_for_model.values.reshape(-1, 1))
                    original_indices = data_for_model.index[outliers == -1]
                    if not original_indices.empty:
                        median_val = df_processed[col].median()
                        df_processed.loc[original_indices, col] = median_val
        self.df = df_processed
        return self.df

    def get_processed_df(self):
        return self.df
    