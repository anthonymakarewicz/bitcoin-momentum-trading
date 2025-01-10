import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis, mstats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformer = PowerTransformer(method='yeo-johnson')
        self._transform_kind = None 

    def set_output(self, transform=None):
        self._transform_kind = transform
        return self

    def fit(self, X, y=None):
        self.transformer.fit(X)
        return self

    def transform(self, X):
        transformed = self.transformer.transform(X)
        if self._transform_kind == 'pandas':
            if hasattr(X, "columns"):
                return pd.DataFrame(transformed, columns=X.columns, index=X.index)
            else:
                return pd.DataFrame(transformed)
        else:
            return transformed
        
    def get_feature_names_out(self):
        pass
    

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=[0.01, 0.01]):
        self.limits = limits
        self._transform_kind = None

    def set_output(self, transform=None):
        self._transform_kind = transform
        return self

    def fit(self, X, y=None):
        # Winsorizer does not need to learn any parameters from X
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            X_values = X.values
            col_names = X.columns
            idx = X.index
        else:
            X_values = X
            col_names = None
            idx = None

        winsorized = np.apply_along_axis(
            lambda col: mstats.winsorize(col, limits=self.limits),
            axis=0,
            arr=X_values
        )

        if self._transform_kind == "pandas":
            if col_names is not None:
                return pd.DataFrame(winsorized, columns=col_names, index=idx)
            else:
                return pd.DataFrame(winsorized)
        else:
            return winsorized
        
    def get_feature_names_out(self):
        pass
        

class DataProcessor:
    def __init__(self, 
                 include_scaling=None,
                 skew_threshold=2, 
                 kurtosis_threshold=10, 
                 winsor_limits=[0.01, 0.01]):
        self.include_scaling = include_scaling or []
        self.skew_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.winsor_limits = winsor_limits
        
        self.preprocessing_pipeline = None
        self.final_pipeline = None

    def _categorize_features(self, X):
        categories = {
            'yeo_johnson_only': [],
            'winsorizing_only': [],
            'yeo_johnson_winsorizing': [],
            'none': []
        }

        for col in X.columns:
            col_skewness = skew(X[col])
            col_kurtosis = kurtosis(X[col], fisher=False)

            if (col_skewness > self.skew_threshold) & (col_kurtosis > self.kurtosis_threshold):
                categories['yeo_johnson_winsorizing'].append(col)
            elif col_skewness > self.skew_threshold:
                categories['yeo_johnson_only'].append(col)
            elif col_kurtosis > self.kurtosis_threshold:
                categories['winsorizing_only'].append(col)
            else:
                categories['none'].append(col)
                
        return categories

    def _build_pipeline(self, columns, categories):
        preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('yeo_johnson_only', YeoJohnsonTransformer(), 
                categories['yeo_johnson_only']),
        
                ('winsorizing_only', Winsorizer(limits=self.winsor_limits),
                categories['winsorizing_only']),
                
                ('yeo_johnson_winsorizing', Pipeline([
                    ('yeo_johnson', YeoJohnsonTransformer()),
                    ('winsor', Winsorizer(limits=self.winsor_limits))
                ]), categories['yeo_johnson_winsorizing']),

                ('none', 'passthrough', categories['none'])
            ], 
            n_jobs=-1
        )
        
        include_scaling = [i for i, col in enumerate(columns) if col in self.include_scaling]
        
        scaling_transformer = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), include_scaling)
            ],
            remainder='passthrough', verbose_feature_names_out=False
        )
        
        final_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('scaling', scaling_transformer)
        ])

        return final_pipeline

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        columns = X_df.columns
        categories = self._categorize_features(X_df)
        self.final_pipeline = self._build_pipeline(columns, categories)
        self.final_pipeline.fit(X_df, y)
        return self

    def transform(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        return pd.DataFrame(self.final_pipeline.transform(X_df), index=X_df.index, columns=X_df.columns)

    def fit_transform(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.fit(X_df, y)
        return pd.DataFrame(self.final_pipeline.transform(X_df), index=X_df.index, columns=X_df.columns)