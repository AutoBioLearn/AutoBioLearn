from functools import wraps
import pandas as pd

def requires_dataset(func):
    def wrapper(self, *args, **kwargs):
        dataset = getattr(self, 'dataset', None) or getattr(getattr(self, 'data_processor', None), 'dataset', None)        
        if dataset is None:
            raise ValueError("Dataset is not initialized.")
        return func(self, *args, **kwargs)
    return wrapper



def apply_per_section(method):
    @wraps(method)
    def wrapper(self, df: pd.DataFrame, *args, **kwargs):
        # Verifica se o DataFrame possui MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            results = {}
            # Itera por cada seção do MultiIndex
            for keys, section in df.groupby(level=list(range(df.index.nlevels))):
                print(f"\nExecutando {method.__name__} para seção com índice: {keys}")
                # Executa o método para a seção e armazena o resultado
                result = method(self, section, *args, **kwargs)
                results[keys] = result
            return results
        else:
            print(f"O DataFrame não possui MultiIndex. Executando {method.__name__} no DataFrame completo.")
            return method(self, df, *args, **kwargs)
    return wrapper

def apply_per_grouping(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):        
        if self.data_processor.dataset.get_has_many_header():
            if "section" not in kwargs:
                results = {}             
                for section in self.data_processor.dataset.get_sections():
                    print(section)
                    result = method(self, *args, **kwargs, section=section)
                    if result is not None:
                        results[section] = result
                if len(results.keys()) > 0:
                    return results
                else:
                    return
            else:
                print(kwargs["section"]) 
        return method(self, *args, **kwargs)
    return wrapper