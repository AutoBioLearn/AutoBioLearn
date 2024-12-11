import os
import pandas as pd
from pandas import DataFrame
from data_treatment.Dataset import Dataset

class DatasetByFile(Dataset):
    def __init__(self, file_path:str, target: str, delimiter: None, verbose=False, header_size=1):
        df: DataFrame

        header = [i for i in range(header_size)]

        file_extension = os.path.splitext(file_path)[1]

        if file_extension in [".xls",".xlsx"]:
            df = pd.read_excel(file_path, header=header)
        elif file_extension == ".csv":
              df = pd.read_csv(file_path,delimiter= delimiter, header=header)
        elif file_extension == ".txt":
             df = pd.read_csv(file_path, sep=delimiter, header=header)
        else:
            raise TypeError("Not support to this extesion")

        super().__init__(df, target,verbose)