from typing import overload
from pandas import DataFrame
from data_treatment.Dataset import Dataset


class DatasetCustomAnalysis(Dataset):
    def __init__(self, df: DataFrame, groups:dict, verbose=False):
        super().__init__(df,"",verbose)
        self._sections_name = list(groups.keys())
        self._has_many_header = len(self._sections_name) > 0
        self.__groups = groups
    
    def get_X(self, section:str= None)->DataFrame:
        cols_names =self.__groups[section]["x_cols_names"]
       
        numeric_cols = [cname for cname in cols_names if self._data[cname].dtype in self._typesToX()]
        X = self._data[numeric_cols].copy()
        target = self.__groups[section]["y_col_name"]

        if target not in numeric_cols:           
            return X 
        else:
            return X.drop([target],axis=1,inplace=False)
        
    
    def _set_sections(self):
        pass

    def get_Y(self, section:str = None)-> DataFrame:
        return self._get_Y(self._data,self.__groups[section]["y_col_name"])
    
    def drop_section(self,sections: list[str]):
        for section in sections:
            del self.__groups[section]
            self._sections_name.remove(section)