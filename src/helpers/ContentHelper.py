import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

class ContentHelper(object):

    @staticmethod
    def convert_datetime(df: DataFrame,cols):       
        if cols is not None:
            cols_treament = cols

            if type(cols) is dict:
                cols_treament = cols.items()

            for col in cols_treament:    
                if type(col) is str:            
                    df[col] = pd.to_datetime(df[col],'ignore').astype('int64')
                else:
                    df[col[0]] = pd.to_datetime(df[col[0]],'ignore',format=col[1]).astype('int64')
    
    @staticmethod
    def try_convert_object_values(df: DataFrame):
        """
        Try execute this after run convert_datetime
        """ 
        ContentHelper.convert_cols_values(df, df.select_dtypes("O").columns)   
    
    @staticmethod
    def convert_cols_values(df: DataFrame, columns=list):
        """
        Try execute this after run convert_datetime
        """             
        for col in columns:
            try:
                series = df[col]
                df[col]=pd.Series(LabelEncoder().fit_transform(series[series.notnull()]), index=series[series.notnull()].index)
            except:
                pass
    
    @staticmethod
    def const_axis_column(): 
        return 1
    
    @staticmethod
    def const_axis_row(): 
        return 0