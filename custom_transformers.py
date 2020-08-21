from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import datetime


class FeatureUnionToDF(BaseEstimator, TransformerMixin):
    """
    Function that converts results of FeatureUnion transformer into DataFrame and restores column names
    of merged DataFrames (based on last step of each merged pipeline)

    Args:
        merged_pipelines: list of pipelines which results' were merged by FeatureUnion transformer
    Returns:
        pd.DataFrame
    """
    
    def __init__(self, merged_pipelines):
        self.__pipelines = merged_pipelines
        self.__col_names = []

    def fit(self, X, y=None):
        self.__col_names = []
        for pipeline in self.__pipelines:
            for feature in pipeline[-1].get_feature_names():
                self.__col_names.append(feature)

        return self

    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.__col_names)


class LogTransformerDF(BaseEstimator, TransformerMixin):
    """
    Custom log transformer for specified columns in DataFrame

    Args:
        features: list of features which has to be log-transformed (np.log)
    Returns:
        pd.DataFrame with transformed features (suffix: '_log') and dropped original values
    """
    
    def __init__(self, features):
        self.__transform_features = features
        self.__aggregates = {}
        self.__features = []

    def fit(self, X, y=None):
        self.__features = X.columns.tolist()

        for feature in self.__transform_features:
            self.__aggregates[feature] = X[feature].min()
        return self

    def get_feature_names(self):
        return self.__features

    def transform(self, X, y=None):
        for feature in self.__transform_features:
            X[feature + '_log'] = np.log(X[feature] - self.__aggregates[feature] + 1)
            X = X.drop(feature, axis=1)

        self.__features = X.columns.tolist()

        return X
    

class FeatureSelectorDF(BaseEstimator, TransformerMixin):
    """
    Simple feature selector that takes DataFrame as an input and returns pd.DataFrame aswell
    
    Args:
        feature_names - list of feature names to keep in DataFrame
    Returns:
        Dataframe with selected features
    """
    
    def __init__(self, feature_names):
        self._feature_names = feature_names
    
    def fit(self, X, y = None):
        return self 
    
    def get_feature_names(self):
        return self._feature_names
    
    
    def transform(self, X, y=None):
        return pd.DataFrame(X[self._feature_names])
    
    
class SimpleImputerDF(BaseEstimator, TransformerMixin):
    """
    Simple imputer with constant strategy that Imputes all missing data in whole DataFrame.
    
    Args:
        strategy: imputation's strategy
            constant: fill column with fixed value
        fill_value: value that replaces missing values
    Returns:
        pd.DataFrame with imputed values
    """
    
    def __init__(self, strategy='constant', fill_value='unknown'):
        self.__strategy = strategy
        self.__fill_value = fill_value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.__strategy == 'constant':
            for col in X.columns:
                if X[col].dtype == object:
                    X[col] = X[col].fillna(self.__fill_value)
            return X
        else:
            raise NotImplementedError("That strategy is not implemented yet")
    
    
class LabelEncoderDF(BaseEstimator, TransformerMixin):
    """
    Label encoder for a DataFrame with category variables. Mapping (as dict) has to be passed as argument
    
    Args:
        mapping_dict: Dictionary with specified mappings (label encoding) for each feature/column
    Returns:
        pd.DataFrame with transformed features
    """
    
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict 
    
    def fit(self, X, y=None):
        self.__input_features = X.columns.tolist()
        return self 
    
    def get_feature_names(self):
        return self.__input_features
    
    def transform(self, X, y=None):
        for col in self.mapping_dict.keys():
            col_dict = self.mapping_dict[col]
            #TODO: extend dictionary with unseen labels. Don't put them in one class -1
            #X[col] = X[col].apply(lambda x: col_dict[x] if x in col_dict.keys() else -1)
            X[col] = X[col].map(col_dict)
        return X
    

class OneHotEncoderToDF(BaseEstimator, TransformerMixin):
    """
    Function takes results from sklearn.preprocessing.OneHotEncoder (scipy.matrix) and convert them to 
    pd.DataFrame and recreate feature/column names after one-hot-encoding
    
    Args:
        feature_names: list of feature names that were one-hot-encoded
        one_hot_encoder: one-hot-encoder that encoded original features (usually previous step in pipeline)
    Returns:
        pd.DataFrame with one-hot-encoded features with column names that specifies original column name and class
    """
    
    def __init__(self, feature_names, one_hot_encoder):
        self.__feature_names = feature_names
        self.__one_hot_encoder = one_hot_encoder
    
    def fit(self, X, y=None):
        self.__categories = self.__one_hot_encoder.categories_
        self.__col_names = []
        for feature, categories in zip(self.__feature_names, self.__categories):
            for category in categories:
                self.__col_names.append(feature + '_' + category.replace(' ', '_'))
                
        return self
    
    def get_feature_names(self):
        return self.__col_names
    
    def transform(self, X, y=None):
        df_X = pd.DataFrame(X.toarray(), columns=self.__col_names)
        
        return df_X
    
    
class NumericalFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering for numerical features that bases on pd.DataFrame
    
    Args:
        None
    Returns:
        pd.DataFrame with new features and dropped original ones.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.__input_features = X.columns.tolist() 
        return self
    
    def get_feature_names(self):
        return self.__input_features
    
    def transform(self, X, y=None):
        current_year = datetime.datetime.now().year
        
        X['Age'] = current_year - X['Year_of_birth']
        X = X.drop('Year_of_birth', axis=1)
        
        X['Years_of_experience'] = current_year - X['Year_of_recruitment']
        X = X.drop('Year_of_recruitment', axis=1)
        
        self.__input_features = X.columns.tolist()

        return X
    

class TargetEncoderDF(BaseEstimator, TransformerMixin):
    """
    Custom target encoder - converts categorical features into discreete distributed numerical feature.
    Function takes pd.DataFrame as an input.
    
    Args:
        features: list of features to encode with target encoder
        target_col: column name of target variable
        strategy: strategy for target encoding
            'mean': calculate mean of target variable per each class
    Returns:
        pd.DataFrame with encoded features (original values are replaced)
    """
    
    def __init__(self, features, target_col, strategy='mean'):
        self.__features = features
        self.__target_col = target_col
        self.__strategy = strategy
        self.__aggregates = {}
    
    def fit(self, X, y=None):
        self.__output_features = self.__features
        
        if self.__strategy == 'mean':
            for feature in self.__features:
                self.__aggregates[feature] = X.groupby(feature)[self.__target_col].agg(mean=np.mean)['mean'].to_dict()
        else:
            raise NotImplementedError("That strategy is not implemented yet")

        return self
    
    def get_feature_names(self):
        return self.__output_features
    
    def transform(self, X, y=None):

        for feature in self.__features:
            X[feature] = X[feature].apply(lambda x: self.__aggregates[feature][x])
        
        return X[self.__output_features]
