from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing_extensions import deprecated
from matplotlib import pyplot as plt
from data_treatment import DataProcessor, DatasetByFile, DatasetByWeb

import pandas as pd
from decorators import apply_per_grouping, requires_dataset
class AutoBioLearn(ABC):

    def __init__(self) -> None:
        self._models_executed = []      

    def load_dataset(self, data_processor: DataProcessor):
        if not hasattr(self, 'data_processor'):
            self.data_processor = data_processor  
    
    def load_dataset_by_file(self, file_path: str,target: str,delimiter: str = None, header_size:int=1):
        dataset= DatasetByFile(file_path=file_path,target=target,delimiter=delimiter, header_size= header_size)
        data_processor = DataProcessor(dataset)
        self.load_dataset(data_processor)
            
    def load_dataset_by_web(self, url: str,target: str, header_size:int=1):
        dataset= DatasetByWeb(url= url,target=target, header_size= header_size)
        data_processor = DataProcessor(dataset)
        self.load_dataset(data_processor)  

    @requires_dataset
    def perform_eda(self, path_to_save_report=None):        
        self.data_processor.dataset.perform_eda(path_to_save_report=path_to_save_report)

    @requires_dataset
    def plot_heatmap(self, show_values = False, remove_repetead_value = False,fig_size= (0,0), section:str=None):
        self.data_processor.dataset.plot_heatmap(show_values=show_values,remove_repetead_value=remove_repetead_value,fig_size=fig_size,section=section)    
    
    @requires_dataset
    def plot_pairplot(self, cols:list[str] = None, height=2.5,section:str = None):
        self.data_processor.dataset.plot_pairplot(cols=cols, height= height,section=section)

    @requires_dataset
    def encode_categorical(self, cols:list[str] = [], parallel: bool = False):
        def process_column(col):           
            self.data_processor.encode_categorical([col])
            
        if parallel:
            with ThreadPoolExecutor() as executor:
                executor.map(process_column, cols)
        else:
            self.data_processor.encode_categorical(cols)

    @requires_dataset
    def drop_cols_na(self, percent=30.0, section: str=None):
        self.data_processor.drop_cols_na(percent,section= section)

    @requires_dataset   
    def drop_rows_na(self, percent=10.0, section: str=None):
        self.data_processor.drop_rows_na(percent,section= section)

    @requires_dataset
    @apply_per_grouping
    def show_cols_na(self, section: str=None):
        self.data_processor.show_cols_na(section= section)
    
    @requires_dataset
    @apply_per_grouping  
    def show_rows_na(self, section: str=None):       
        self.data_processor.show_rows_na(section= section)

    @requires_dataset
    @apply_per_grouping  
    def plot_cols_na(self, value="percent", section: str=None):
        self.data_processor.plot_cols_na(value=value,section=section)

    @requires_dataset
    @apply_per_grouping  
    def plot_rows_na(self, value="percent", section: str=None):
        self.data_processor.plot_rows_na(value=value,section=section)

    @requires_dataset
    def remove_cols(self, cols:list[str] = []):
        self.data_processor.remove_cols(cols)

    @requires_dataset
    @apply_per_grouping 
    def remove_duplicates(self, section: str=None):      
        self.data_processor.dataset.remove_duplicates(section= section)
    
    @requires_dataset    
    def drop_section(self, sections: list[str]):      
        self.data_processor.dataset.drop_section(sections)

    @requires_dataset    
    def standardize(self, sections: list[str]):      
        self.data_processor.dataset.drop_section(sections)

    @requires_dataset
    def encode_datetime(self, cols:list[str] = [], cols_levels= 0, parallel: bool = False):
        def process_column(col):               
            self.data_processor.encode_datetime([col], cols_levels= cols_levels)
            
        if parallel:
            with ThreadPoolExecutor() as executor:
                executor.map(process_column, cols)
        else:
            self.data_processor.encode_datetime(cols, cols_levels= cols_levels)

    @requires_dataset    
    def impute_cols_na(self,method="knn", section: str=None):       
        self.data_processor.dataset.impute_cols_na(method=method, section= section)  

    @requires_dataset  
    def plot_outliers(self, cols_per_row=3,section: str=None):
        self.data_processor.dataset.plot_outliers(cols_per_row= cols_per_row,section=section)

    @requires_dataset  
    def remove_outliers(self, method_remove= "limit_method",cols:list[str] = [] ,section: str=None):
        self.data_processor.dataset.remove_outliers(method_remove=method_remove, cols=cols,section=section)

    @abstractmethod
    def execute_models(self, models:list[str]=["xgboost"],  times_repeats:int=10, params={}, section:str=None):
        return   

    @abstractmethod
    def evaluate_models(self, metrics:list[str]=[], section: str = None)-> dict:
        return
    
    @abstractmethod
    def _calculate_metrics(self):
        return
    
    def plot_metrics(self, metrics:list[str]=[],rot=90, figsize=(12,6), fontsize=20, section: str = None ):
        if not hasattr(self, '_metrics'):
            self._calculate_metrics()

        section_metrics = self._metrics
        
        if section is not None and self.data_processor.dataset.get_has_many_header():
            section_metrics = self._metrics[self._metrics["Section"] == section]

        for metric in metrics:                
            df2  = pd.DataFrame({col:vals[metric] for col, vals in section_metrics.groupby("Model")})
            meds = df2.median().sort_values(ascending=False)
            axes = df2[meds.index].boxplot(figsize=figsize, rot=rot, fontsize=fontsize,
                                        #by="Model",
                                        boxprops=dict(linewidth=4, color='cornflowerblue'),
                                        whiskerprops=dict(linewidth=4, color='cornflowerblue'),
                                        medianprops=dict(linewidth=4, color='firebrick'),
                                        capprops=dict(linewidth=4, color='cornflowerblue'),
                                        flierprops=dict(marker='o', markerfacecolor='dimgray',
                                                        markersize=12, markeredgecolor='black'))
            axes.set_ylabel(metric, fontsize=fontsize)
            axes.set_title("")
            axes.get_figure().suptitle('Boxplots of %s metric' % (metric),
                        fontsize=fontsize)
            #axes.get_figure().show()
            plt.show()  
                  
               
#region Deprecated

    @requires_dataset
    @deprecated("Method will be deprecated, consider using generate_data_report")
    def data_analysis(self, path_to_save_report=None):
        self.data_processor.dataset.perform_eda(path_to_save_report=path_to_save_report)

    @requires_dataset
    @deprecated("Method will be deprecated, consider using encode_categorical")
    def convert_categorical_to_numerical(self, cols:list[str] = []):
        self.data_processor.encode_categorical(cols)
    
    @requires_dataset
    @apply_per_grouping
    @deprecated("Method will be deprecated, consider using show_cols_na")
    def print_cols_na(self, section: str=None):
        self.data_processor.show_cols_na(section= section)
    
    @requires_dataset
    @apply_per_grouping
    @deprecated("Method will be deprecated, consider using show_rows_na")  
    def print_rows_na(self, section: str=None):
        self.data_processor.show_rows_na(section= section)

    @requires_dataset
    @deprecated("Method will be deprecated, consider using encode_numerical")  
    def convert_datetime_to_numerical(self, cols:list[str] = [], cols_levels= 0):
        self.data_processor.encode_datetime(cols, cols_levels= cols_levels)     
#endregion