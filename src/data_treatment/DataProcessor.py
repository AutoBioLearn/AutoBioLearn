from data_treatment import Dataset
from decorators.DatasetDecorators import requires_dataset
from helpers.ContentHelper import ContentHelper


class DataProcessor:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @requires_dataset
    def encode_categorical(self, cols):
        self.dataset.encode_categorical(cols)

    @requires_dataset
    def drop_cols_na(self, percent=30.0, section: str=None):
        self.dataset.drop_na(axis=ContentHelper.const_axis_column(), percent=percent, show_dropped=True, section= section)

    @requires_dataset
    def drop_rows_na(self, percent=10.0, section: str=None):        
        self.dataset.drop_na(axis=ContentHelper.const_axis_row(), percent=percent, show_dropped=True, section= section)
    
    @requires_dataset
    def show_cols_na(self, section: str=None):        
        self.dataset.print_na(axis=ContentHelper.const_axis_column(), section= section)
    
    @requires_dataset
    def show_rows_na(self, section: str=None):        
        self.dataset.print_na(axis=ContentHelper.const_axis_row(), section= section)

    @requires_dataset
    def remove_cols(self, cols:list[str] = []):      
        self.dataset.clean_data(cols_to_drop =cols)    
    

    @requires_dataset
    def encode_datetime(self, cols:list[str] = []):  
        self.dataset.clean_data(cols_date=cols)   

    @requires_dataset
    def plot_cols_na(self, value="percent", section: str=None):
        self.dataset.plot_na(axis=ContentHelper.const_axis_column(),value= value , section= section)

    @requires_dataset
    def plot_rows_na(self, value="percent", section: str=None):
        self.dataset.plot_na(axis=ContentHelper.const_axis_row(),value= value , section= section)