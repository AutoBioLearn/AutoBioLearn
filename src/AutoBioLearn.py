from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing_extensions import deprecated
from matplotlib import pyplot as plt
from data_treatment import DataProcessor, DatasetByFile, DatasetByWeb

import pandas as pd
from decorators import apply_per_grouping, requires_dataset

from helpers import XAIHelper, ModelHelper

class AutoBioLearn(ABC):

    def __init__(self) -> None:
        self._models_executed = []
        self._validations_execution = {}
        validation_object =self._get_validation("split")
        self._validations_execution["split"] = {
            'validation': validation_object,
            'num_folds': 0,
            'train_size': 70
        }  

    def get_dataset(self, data_processor: DataProcessor):
        if not hasattr(self, 'data_processor'):
            self.data_processor = data_processor  
    
    def get_dataset_by_file(self, file_path: str,target: str,delimiter: str = None, header_size:int=1):
        dataset= DatasetByFile(file_path=file_path,target=target,delimiter=delimiter, header_size= header_size)
        data_processor = DataProcessor(dataset)
        self.get_dataset(data_processor)
            
    def get_dataset_by_web(self, url: str,target: str, header_size:int=1):
        dataset= DatasetByWeb(url= url,target=target, header_size= header_size)
        data_processor = DataProcessor(dataset)
        self.get_dataset(data_processor)  

    @requires_dataset
    def generate_data_report(self, path_to_save_report=None):        
        self.data_processor.dataset.generate_data_report(path_to_save_report=path_to_save_report)

    @requires_dataset
    def generate_data_heatmap(self, show_values = False, remove_repetead_value = False,fig_size= (0,0), section:str=None):
        self.data_processor.dataset.generate_data_heatmap(show_values=show_values,remove_repetead_value=remove_repetead_value,fig_size=fig_size,section=section)    
    
    @requires_dataset
    def generate_data_pairplot(self, cols:list[str] = None, height=2.5,section:str = None):
        self.data_processor.dataset.generate_data_pairplot(cols=cols, height= height,section=section)

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

    def set_validations(self, validations:list[str]=["split"], params ={}):
        self._validations_execution= {}     

        unique_validations = set(validations)

        for validation in unique_validations:
            validation_object = self._get_validation(validation)
            validation_params = ModelHelper.get_model_params(validation,params)
            self._validations_execution[validation] =  { 'validation': validation_object, 'num_folds': validation_params["num_folds"],'train_size':validation_params["train_size"]}
    
    @abstractmethod
    def _get_validation(self,validation: str):
        return

    @abstractmethod
    def execute_models(self, models:list[str]=["xgboost"],  times_repeats:int=10, params={}, section:str=None):
        return
      
    def _add_model_executed(self ,time: int,validation: str, fold: int,                                                         
                            model_name: str, model,y_pred, y_test, x_test_index, section= None):
        
        instance = {"time":time,
                    "validation":validation,
                    "fold":fold,                                                        
                    "model_name":model_name,
                    "model":model,
                    "y_pred":y_pred,
                    "y_test":y_test,
                    "x_test_index":x_test_index }
        
        if section:
           instance["section"] = section 

        self._models_executed.append(instance)


    def _find_best_hyperparams(self, clf_model,
                            X,
                            y,
                            param_grid,
                            param_sel_obj,
                            num_folds,
                            metric
                          ):  

    # Grid search optimal parameters
        clf_grid = param_sel_obj(clf_model,
                                  param_grid,                                  
                                  cv=num_folds,
                                  scoring = metric
                                  )

    # training model
        clf_grid.fit(X, y)
        return clf_grid.best_params_

    def evaluate_models(self, metrics:list[str]=[], section: str = None)-> dict:
        if not hasattr(self, '_metrics'):
            self._calculate_metrics()

        all_list = {}

        section_metrics = self._metrics

        if section is not None and self.data_processor.dataset.get_has_many_header():
            section_metrics = self._metrics[self._metrics["Section"] == section]

        for metric in metrics:
            all_list[metric] = section_metrics[["Model",metric ]].groupby('Model').describe()      
        
        all_list["complete"] = section_metrics[["Model","Validation","Time_of_execution","Fold"]+ metrics]
        return all_list
    
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

    def generate_shap_analysis(self,**kwargs):
        """
        kwargs use a list to filter by key models to analisys, where each key receives a list of values that will be filtered 
        kwargs params: time, validation, model_name, fold.
        Eg.: fold = [1,2,3]
        """
        models_explained = self._models_executed.copy()

        for key, value in kwargs.items():
            models_explained = filter(lambda x: x[key] in value, models_explained)      
        
        self.__SHAP_analisys = []

        x : pd.DataFrame = None
        if not self.data_processor.dataset.get_has_many_header():
            x = self.data_processor.dataset.get_X()

        def explain_current_model(model_to_explain, x):
            shap_model_analisys = {"time":model_to_explain["time"],
                                        "validation":model_to_explain["validation"],
                                        "fold":model_to_explain["fold"],
                                        "model_name":model_to_explain["model_name"],
                                        "x_test_index": model_to_explain["x_test_index"]
                                    }

            if "section" in model_to_explain:                
                shap_model_analisys["section"] = model_to_explain["section"]
                x = self.data_processor.dataset.get_X(model_to_explain["section"])

            x_to_consolidated = x.iloc[model_to_explain["x_test_index"]]

            explainer_consolidated =  XAIHelper.get_explainer(model=model_to_explain,X=x_to_consolidated)

            shap_values_consolidated = explainer_consolidated.shap_values(x_to_consolidated)
            shap_obj_consolidated    = explainer_consolidated(x_to_consolidated)
            
            expected_value_consolidated = explainer_consolidated.expected_value            

            explainer =   XAIHelper.get_explainer(model=model_to_explain,X=x)

            shap_values = explainer.shap_values(x)
            shap_obj    = explainer(x)

            expected_value = explainer.expected_value
            shap_model_analisys["shap_obj"]= shap_obj
            shap_model_analisys["shap_values"]= shap_values
            shap_model_analisys["expected_value"]=expected_value
            shap_model_analisys["shap_obj_consolidated"]= shap_obj_consolidated
            shap_model_analisys["shap_values_consolidated"]= shap_values_consolidated
            shap_model_analisys["expected_value_consolidated"]=expected_value_consolidated

            return shap_model_analisys


        with ThreadPoolExecutor() as executor:
            future_to_model = [executor.submit(explain_current_model, models_to_execute, x) for models_to_execute in models_explained]

            for future in as_completed(future_to_model):               
                try:                   
                    shap_model_analisys = future.result()
                    self.__SHAP_analisys.append(shap_model_analisys)
                except Exception as e:
                   print(e)
                   pass                  
        
    @apply_per_grouping        
    def plot_shap_analysis(self,register=None,graph_type_global="summary",graph_type_local="force",show_all_features =True,class_index: int =0,**kwargs):
        """
        class_index works only lightgbm models, class_index is max value the number of classes in dataset -1.(Eg.: total class = 3, class_index_max=2)
        kwargs use a list to filter by key models to analisys, where each key receives a list of values that will be filtered 
        kwargs params: time, validation, model_name, fold.
        Eg.: fold = [1,2,3]
        """       

        models_explained = self.__SHAP_analisys.copy()

        kwargs_filtered_models = {key: value for  key, value in kwargs.items() if key not in "graph_params"}

        for key, value in kwargs_filtered_models.items():
            models_explained = list(filter(lambda x: x[key] in value, models_explained)) 
        
        if not self.data_processor.dataset.get_has_many_header():
            X = self.data_processor.dataset.get_X()

        kwargs_filtered_graph= {key: value for  key, value in kwargs.items() if key in "graph_params"}       
 
        for model_explainable in models_explained:
            if "section" in model_explainable:
                X = self.data_processor.dataset.get_X(model_explainable["section"])
                    
            if register is not None:
                expected_value = model_explainable["expected_value"]
                shap_values = model_explainable["shap_values"]

                if model_explainable["model_name"] == ModelHelper.const_lightboost():
                    expected_value = expected_value[class_index]
                    shap_values =  shap_values[class_index]
                
                XAIHelper.get_chart_type_local(graph_type_local,expected_value,shap_values[register],X.iloc[register], \
                                                         kwargs_filtered_graph, show_all_features=show_all_features)

            else:
                shap_values = model_explainable["shap_values"]
                if model_explainable["model_name"] == ModelHelper.const_lightboost() and graph_type_global != "bar":
                    shap_values=  shap_values[class_index]
               
                XAIHelper.get_chart_type_global(graph_type_global,shap_values,X,kwargs_filtered_graph, show_all_features=show_all_features)
    
    @apply_per_grouping
    def plot_shap_analysis_consolidated(self,graph_type="summary",show_all_features =True,class_index=0,**kwargs):
        """
        class_index works only lightgbm models, class_index is max value the number of classes in dataset -1.(Eg.: total class = 3, class_index_max=2)
        kwargs use a list to filter by key models to analisys, where each key receives a list of values that will be filtered 
        kwargs params: time, validation, model_name, fold.
        Eg.: fold = [1,2,3]
        """       

        models_explained = self.__SHAP_analisys.copy()

        kwargs_filtered_models = {key: value for  key, value in kwargs.items() if key not in "graph_params"}

        for key, value in kwargs_filtered_models.items():
            models_explained = list(filter(lambda x: x[key] in value, models_explained)) 
        
        if not self.data_processor.dataset.get_has_many_header():
            X = self.data_processor.dataset.get_X()

        kwargs_filtered_graph= {key: value for  key, value in kwargs.items() if key in "graph_params"}
                  
        models = list(set([x["model_name"] for x in models_explained]))
            
        for model in models:
            print("Model:", model)
            if "section" in kwargs:
                X = self.data_processor.dataset.get_X(kwargs["section"])
            if model == ModelHelper.const_lightboost():
                shap_values, X_test = XAIHelper.get_consolidate_shap_values_lightboost(models_explained, model, X, class_index,None)
                XAIHelper.get_chart_type_global(graph_type,shap_values,X_test,kwargs_filtered_graph, show_all_features=show_all_features)
            else:
                shap_values, X_test = XAIHelper.get_consolidate_shap_values(models_explained, model, X, None)
                XAIHelper.get_chart_type_global(graph_type,shap_values,X_test,kwargs_filtered_graph, show_all_features=show_all_features)
        
                 
               
#region Deprecated

    @requires_dataset
    @deprecated("Method will be deprecated, consider using generate_data_report")
    def data_analysis(self, path_to_save_report=None):
        self.data_processor.dataset.generate_data_report(path_to_save_report=path_to_save_report)

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

    @deprecated("Method will be deprecated, consider using generate_shap_analysis")  
    def run_xai_analysis(self,**kwargs):
        self.generate_shap_analysis(**kwargs)

    @apply_per_grouping
    @deprecated("Method will be deprecated, consider using plot_shap_analysis")        
    def plot_xai_analysis(self,index_to_filter=None,consolidated= False,graph_type_global="summary",graph_type_local="force",show_all_features =True,class_index=0,**kwargs):
        self.plot_shap_analysis(index_to_filter, consolidated,graph_type_global,graph_type_local,show_all_features,class_index,**kwargs)
#endregion