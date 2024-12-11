import pandas as pd
from pandas import DataFrame, MultiIndex

class DatasetHelper(object):

    @staticmethod
    def find_outliers_IQR(df: DataFrame):
        #https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/
        #https://hersanyagci.medium.com/detecting-and-handling-outliers-with-pandas-7adbfcd5cad8        
      
        limit_min,limit_max = DatasetHelper.find_outliers_limit_IQR(df)

        outliers = df[((df<limit_min) | (df>limit_max))]

        return outliers
    
    @staticmethod
    def find_outliers_limit_IQR(df: DataFrame):
        
        q1=df.quantile(0.25)

        q3=df.quantile(0.75)

        IQR=q3-q1

        return ((q1-1.5*IQR),(q3+1.5*IQR))

    @staticmethod
    def find_columns_with_outliers(df: DataFrame):
        return DatasetHelper.find_outliers_IQR(df).dropna(axis=1,how='all').columns.values

    @staticmethod
    def normalize_columns_name(df: DataFrame):
        dict_of_str = {
            '<=': 'lte ',
            '>=': 'gte ',
            '<': 'lt ',
            '>': 'gt ',
            '=': 'eq ',
            ',': ' |'}

        nlevels = df.columns.nlevels 
   
        if isinstance(df.columns, pd.MultiIndex):
            for i in range(nlevels):
                for key,value in dict_of_str.items():
                    df.columns = df.columns.set_levels(df.columns.levels[i].str.replace(key, value), level=i)
        else:
            for key,value in dict_of_str.items():
                df.columns = df.columns.str.replace(key, value)       