import os
import pandas as pd
from pandas import DataFrame
from data_treatment.Dataset import Dataset

class DatasetByFile(Dataset):
    def __init__(self,
                 file_path:str,
                 delimiter: None,
                 target: str|None=None,
                 verbose=False,
                 header_size=1,
                 index_col=None):
        df: DataFrame

        if header_size == 1:
            header = 0
        else:
            header = [i for i in range(header_size)]

        file_extension = os.path.splitext(file_path)[1]

        if file_extension in [".xls",".xlsx"]:
            df = pd.read_excel(file_path,
                               header=header,
                               index_col=index_col)
        elif file_extension == ".csv":
            df = pd.read_csv(file_path,
                             delimiter= delimiter,
                             header=header,
                             index_col=index_col)
        elif file_extension == ".txt":
            df = pd.read_csv(file_path, 
                             sep=delimiter,
                             header=header,
                             index_col=index_col)
        elif file_extension in [".odf", ".ods", ".odt"]:
            df = pd.read_excel(file_path,
                               header=header,
                               engine="odf",
                               index_col=index_col)
        else:
            raise TypeError("Not support to this extesion")

        super().__init__(df, target,verbose)
