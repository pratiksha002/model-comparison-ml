import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

class SmartPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, add_polynomial=False, degree=2, handle_outliers=False, handle_skew=True, skew_threshold=0.75, feature_selection=True, variance_threshold=0.01):
        self.add_polynomial = add_polynomial
        self.degree = degree
        self.handle_outliers = handle_outliers
        self.handle_skew = handle_skew
        self.skew_threshold = skew_threshold
        self.feature_selection = feature_selection
        self.variance_threshold = variance_threshold

    def fit(self, x, y=None):
        x = pd.DataFrame(x)

        #column detection
        self.num_cols = x.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = x.select_dtypes(exclude=np.number).columns.tolist()

        #imputer
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")

        self.global_mean = y.mean() if y is not None else 0

        #fit the imputers
        x_num = self.num_imputer.fit_transform(x[self.num_cols])
        x_cat = self.cat_imputer.fit_transform(x[self.cat_cols])

        #skew detection 
        if self.handle_skew:
            skewness = pd.DataFrame(x_num, columns=self.num_cols).skew()
            self.skewed_features = skewness[skewness > self.skew_threshold].index.tolist()
        else:
            self.skewed_features = []

        # Target Encoding
        if y is not None:
            self.target_encoding_maps = {}
            x_cat_df = pd.DataFrame(x_cat, columns=self.cat_cols)

            for col in self.cat_cols:
                temp = pd.DataFrame({col: x_cat_df[col], "target": y})
                self.target_encoding_maps[col] = temp.groupby(col)["target"].mean().to_dict()

        #transform the skewed features
        if self.handle_skew:
            x_num = self._handle_skew(x_num)

        #scaling
        self.scaler = StandardScaler()
        self.scaler.fit(x_num)

        #polynomial
        if self.add_polynomial:
            self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
            x_num = self.poly.fit_transform(x_num)

        #encode (kept but not used after target encoding)
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder.fit(x_cat)

        #combine (using target encoding instead of one-hot)
        x_cat_df = pd.DataFrame(x_cat, columns=self.cat_cols)

        for col in self.cat_cols:
            x_cat_df[col] = x_cat_df[col].map(self.target_encoding_maps[col])
            x_cat_df[col] = x_cat_df[col].fillna(self.global_mean)
        x_cat_df = x_cat_df.fillna(self.global_mean)
        x_cat = x_cat_df.values
        x_full = np.hstack([x_num, x_cat])
        x_full = np.nan_to_num(x_full, nan=self.global_mean)


        #feature selection
        if self.feature_selection:
            self.selector = VarianceThreshold(threshold=self.variance_threshold)
            self.selector.fit(x_full)

        return self

    def transform(self, x):
        x = pd.DataFrame(x)

        #impute
        x_num = self.num_imputer.transform(x[self.num_cols])
        x_cat = self.cat_imputer.transform(x[self.cat_cols])

        #outliers
        if self.handle_outliers:
            x_num  = self._clip_outliers(x_num)

        #skew
        if self.handle_skew:
            x_num = self._handle_skew(x_num)

        #scale
        x_num = self.scaler.transform(x_num)

        #polynomial
        if self.add_polynomial:
            x_num = self.poly.transform(x_num)

        #target encoding
        x_cat_df = pd.DataFrame(x_cat, columns=self.cat_cols)

        for col in self.cat_cols:
           x_cat_df[col] = x_cat_df[col].map(self.target_encoding_maps[col])

           x_cat_df[col].fillna(self.global_mean)
        x_cat_df = x_cat_df.fillna(self.global_mean)
        x_cat = x_cat_df.values

        #combine
        x_full = np.hstack([x_num, x_cat])
        x_full = np.nan_to_num(x_full, nan=self.global_mean)

        #feature selection
        if self.feature_selection:
            x_full = self.selector.transform(x_full)

        return x_full
    
    def _clip_outliers(self, x):
        Q1 = np.percentile(x, 25, axis=0)
        Q3 = np.percentile(x, 75, axis=0)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return np.clip(x, lower, upper)
    
    def _handle_skew(self, x):
        if not self.skewed_features:
            return x
        
        x_df = pd.DataFrame(x, columns=self.num_cols)

        for col in self.skewed_features:
            x_df[col] = np.log1p(x_df[col])

        return x_df.values
    
    def get_feature_names_out(self):
        num_features = self.num_cols

        if self.add_polynomial:
            num_features = self.poly.get_feature_names_out(self.num_cols)

        # simplified feature names for target encoding
        cat_features = self.cat_cols

        all_features =  np.concatenate([num_features, cat_features])

        if self.feature_selection:
            mask = self.selector.get_support()
            all_features = all_features[mask]

        return all_features