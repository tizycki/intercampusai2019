from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class CustomFeatureUnionToDataFrame(BaseEstimator, TransformerMixin):
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


class LogTransformer(BaseEstimator, TransformerMixin):
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
            X[feature + '_log'] = X[feature].apply(lambda x: np.log(x - self.__aggregates[feature] + 1))
            X = X.drop(feature, axis=1)

        self.__features = X.columns.tolist()

        return X