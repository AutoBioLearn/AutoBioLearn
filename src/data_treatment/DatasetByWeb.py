import requests
from data_treatment.Dataset import Dataset
import pandas as pd


class DatasetByWeb(Dataset):
     def __init__(self, url:str, target: str, verbose= False, header_size=1):      

        header = [i for i in range(header_size)]

        response = requests.get(url)
        data_json = response.json()
        df = pd.read_json(data_json, header=header)     

        super().__init__(df, target,verbose)
