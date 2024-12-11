from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame
from ydata_profiling import ProfileReport

from scipy import stats
from scipy.stats.mstats import winsorize

import numpy as np

from helpers.DatasetHelper import DatasetHelper
from helpers.ContentHelper import ContentHelper

from sklearn.impute import KNNImputer, SimpleImputer

import seaborn as sns
class Dataset:
    
    def __init__(self, original_data: DataFrame, target: str, verbose= False):        
      
        DatasetHelper.normalize_columns_name(original_data)

        self.__original_data = original_data.copy(deep=True)
        self._data = original_data.copy(deep=True)      
        self._has_many_header = isinstance(original_data.columns, pd.MultiIndex)
        self.__target = target
        self._sections = {}
        self._sections_name = []

        self._set_sections()    

        if verbose:
            print("Details of your Database")    
            #region Data balacing        

            _classes = self._data[self.__target].value_counts(normalize=True) * 100

            print("Percent of class:")
            print(_classes)

            #endregion

            #region Outliers

            self.__cols_outliers = DatasetHelper.find_columns_with_outliers(self._data)
            if any( self.__cols_outliers):
                print("Columns with outliers:")
                print("\n".join(self.__cols_outliers))

            #endregion
                
    def generate_data_report(self,path_to_save_report=None):
        profile = ProfileReport(self._data, title="Data Analysis")
        report = profile.to_html()   
        
        if path_to_save_report is not None:
            f = open(f"{path_to_save_report}.html", "w")
            f.write(report)
            f.close()
        
        return profile.to_notebook_iframe()

    def _typesToX(self):
        return ['int64', 'float64','int32']   
    
    def get_has_many_header(self)-> bool:
        return self._has_many_header
    
    def get_sections(self)-> list:
        return self._sections_name

    def remove_duplicates(self, use_original_data: False, section:str=None):
        if not self._has_many_header and bool(self._sections):
            if use_original_data:
                self._data = self.__original_data.drop_duplicates(ignore_index=True)
            else:
                self._data = self._data.drop_duplicates(ignore_index=True)
        else:
            self._sections[section]= self._sections[section].drop_duplicates(ignore_index=True)

    def __count_na(self, df,axis=1):
        neg_axis = 1 - axis
        count   = df.isna().sum(axis=neg_axis)
        percent = df.isna().mean(axis=neg_axis) * 100
        df_na   = pd.DataFrame({'percent':percent, 'count':count})
        index_name = 'column' if axis==ContentHelper.const_axis_column() else 'index'
        df_na.index.name = index_name
        df_na.sort_values(by='count', ascending=False, inplace=True)
        return df_na

    def __print_na(self, df, axis=1):
        df_na = self.__count_na(df,axis=axis)
        #
        max_rows = pd.get_option('display.max_rows', None)
        pd.set_option('display.max_rows', None)
        print(df_na)
        pd.set_option('display.max_rows', max_rows)

    def print_na(self, axis=1, section:str=None):       
        if not self._has_many_header or not bool(self._sections):
           self.__print_na(self._data,axis)
        else:           
            self.__print_na(self._sections[section],axis=axis)              

    def plot_na(self, axis=1, value="percent",section:str=None):
        df_na = pd.DataFrame
        y_label_options = {'percent':'Percent of Missing Data', 'count': 'Count of Missing Data'}
        if not self._has_many_header or not bool(self._sections):
            df_na = self.__count_na(self._data,axis)
        else:           
            df_na = self.__count_na(self._sections[section],axis=axis)
        
        if value not in ["percent","count"]:
            raise ValueError("value only permits {'percent','count'} ")

        df_na[value].plot(kind='bar', figsize=(8, 6), rot=90)
        plt.ylabel(y_label_options[value])      

        plt.show()    
    
    def __drop_na(self, df ,axis=0, percent=30.0, show_dropped=True):
        df_na = self.__count_na(df=df,axis=axis)
        to_drop = df_na[df_na['percent'] > percent].index
        if show_dropped:
            print(to_drop)
            
        return df.drop(labels=to_drop, axis=axis)

    def drop_na(self, axis=0, percent=30.0, show_dropped=True, section:str=None):
        if not self._has_many_header or not bool(self._sections):
           self._data = self.__drop_na(self._data, axis,percent,show_dropped)
        else:
            self._sections[section] = self.__drop_na(self._sections[section], axis,percent,show_dropped)

    def remove_outliers(self, method_remove= "limit_method", use_original_data= False):
        if method_remove is None:
            raise AttributeError("method_remove is not null")
        
        if use_original_data:
            self._data = self.__original_data
       
        for col in self.__cols_outliers:

            if method_remove == "limit_method":

                min_limit,max_limit =DatasetHelper.find_outliers_limit_IQR(self._data[col])
                
                #add +1 for remove all outliers, because if have a a big difference between limits and values outliers, still go have new outliers 
                percentile_min = (stats.percentileofscore(self._data[col], min_limit)+1)/100
                percentile_max = (100-stats.percentileofscore(self._data[col], max_limit)+1)/100
                self._data[col] = winsorize(self._data[col], limits=[percentile_min, percentile_max])

            elif method_remove == 'log_transformation':

                self._data[col] = np.log(self._data[col])

            elif method_remove == 'mean_value':

                mean = np.mean(self._data[col])
                df_aux= DatasetHelper.find_outliers_IQR(self._data[col]).dropna(axis=0,how='all')
                self._data.loc[self._data[col].isin(df_aux),col] = mean
            else:
                raise AttributeError("method_remove not exists")
   

    def clean_data(self,cols_to_drop:list[str] = [],cols_date:list[str] = [], try_convert_values= False, use_original_data= False):
        if use_original_data:
            self._data = self.__original_data
        
        if cols_to_drop is not None and any(cols_to_drop):
            if self._has_many_header and bool(self._sections):
                cols_filtered = [self.__find_multiindex(col) for col in cols_to_drop]
                for section in self._sections_name:
                    cols_section = [col[1] for col in cols_filtered if col[0] == section]
                    self._sections[section].drop(cols_section,axis=1,inplace=True)
            else:
                self._data.drop(cols_to_drop,axis=1,inplace=True)
            

        if cols_date is not None and any(cols_date):
            if self._has_many_header and bool(self._sections):
                cols_filtered={}             
                for i in range(len(cols_date)):
                    if isinstance(cols_date[i], str):
                        col_path = self.__find_multiindex(cols_date[i])                    
                    else:
                        col_path = self.__find_multiindex(cols_date[i][0])
                    if col_path[0] in cols_filtered:
                        cols_filtered[col_path[0]].append(cols_date[i])
                    else:
                        cols_filtered[col_path[0]] = list()
                    
                    for section in cols_filtered.keys():
                        ContentHelper.convert_datetime(self._sections[section],cols_filtered[section])
            else:
                ContentHelper.convert_datetime(self._data,cols_date)

        if try_convert_values:
            ContentHelper.try_convert_object_values(self._data)  

    def encode_categorical(self, cols:list[str] = [""]):
        if self._has_many_header and bool(self._sections):
            cols_filtered = [self.__find_multiindex(col) for col in cols]
            for section in self._sections_name:
                cols_section = [col[1] for col in cols_filtered if col[0] == section]
                ContentHelper.convert_cols_values(self._sections[section],cols_section)
        else:
            ContentHelper.convert_cols_values(self._data,cols)
    
    def get_X(self, section: str= None)->DataFrame:
        if section is None and not self._has_many_header:
            numeric_cols = [cname for cname in self._data.columns if self._data[cname].dtype in self._typesToX()]
            X = self._data[numeric_cols].copy()
        else:
            numeric_cols = [cname for cname in self._sections[section].columns if self._sections[section][cname].dtype in self._typesToX()]
            X = self._sections[section][numeric_cols].copy()
        
        if self.__target not in numeric_cols:           
            return X 
        else:
            return X.drop([self.__target],axis=1,inplace=False)
    
    def get_Y(self, section: str= None)->DataFrame:
        if self._has_many_header:
            return self._get_Y(self._sections[section],self.__target)
        else:   
            return self._get_Y(self._data,self.__target)
    
    def _get_Y(self, df,target)->DataFrame:
        if df[target].dtype not in self._typesToX():
            ContentHelper.convert_cols_values(df,[target])
            
        return df[target]
    
    def __impute_cols_na(self, df ,method="knn", n_neighbors=5):
        if method == "knn":
            imputer = KNNImputer(n_neighbors=n_neighbors)            
        else:
           imputer = SimpleImputer(fill_value=np.nan, startegy=method)
        
        df_return =  pd.DataFrame(imputer.fit_transform(df))
        df_return.columns = df.columns
        df_return.index = df.index
        return df_return

    def impute_cols_na(self, method="knn", n_neighbors=5, section: str=None):
        if not self._has_many_header or not bool(self._sections):
            self._data=self.__impute_cols_na(self._data, method, n_neighbors) 
        else:
            self._sections[section]=self.__impute_cols_na(self._sections[section], method, n_neighbors) 

    def drop_section(self,sections: list[str]):
        for section in sections:
            del self._sections[section]
            self._sections_name.remove(section)

    def __find_multiindex(self, col):      
        for idx in self._data.columns:
            if col in idx:
                return idx
        return None    

    def _set_sections(self): 
        if self._has_many_header:
            mi_target = np.array(self.__find_multiindex(self.__target))        
            
            for col in self._data.columns.get_level_values(0).unique():
                if col in mi_target:
                    cols_to_filter = self._data.columns[self._data.columns.get_level_values(0) == col].values.tolist()
                else:
                    cols_to_filter = self._data.columns[self._data.columns.get_level_values(0) == col].values.tolist()+[mi_target]
                
                self._sections[col]=self._data[cols_to_filter].droplevel(0,1)
                self._sections_name.append(col)

    
    def generate_data_heatmap(self, show_values = False, remove_repetead_value = False, fig_size= (0,0), section:str=None):
        if remove_repetead_value:
            self.__generate_data_heatmap_custom(show_values=show_values,fig_size=fig_size,section=section)
        else:
            if fig_size != (0,0):
                _, ax = plt.subplots(figsize=fig_size)
                sns.heatmap(self.get_X(section).corr(), ax= ax,vmin=-1, vmax=1, annot=show_values,  linewidths=.5, fmt=".2f")
            else:
                sns.heatmap(self.get_X(section).corr(), vmin=-1, vmax=1, annot=show_values,  linewidths=.5, fmt=".2f")
    
    def __generate_data_heatmap_custom(self, show_values = False, fig_size= (0,0),section:str=None):
        X_corr = self.get_X(section).corr()
        mask = np.triu(X_corr)
        if fig_size != (0,0):
            _, ax = plt.subplots(figsize=fig_size)

            sns.heatmap(X_corr, mask=mask, ax= ax,
            annot=show_values, fmt=".2f", linewidths=0.5,
            vmin=-1, vmax=1)
        else:           
            sns.heatmap(X_corr, mask=mask,
                annot=show_values, fmt=".2f", linewidths=0.5,
                vmin=-1, vmax=1)
      
    def generate_data_pairplot(self, cols:list[str] = None, height=2.5,section:str = None):
        
        if section is not None:
            df = self._sections[section]
        else:
            df = self._data

        if cols is not None and len(cols) > 0:
            df = df[cols+[self.__target]]
        
        sns.pairplot(df, height=height, hue=self.__target, palette='coolwarm')
        #sns.pairplot(df, height=height, palette='tab10',)