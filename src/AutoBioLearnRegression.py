from concurrent.futures import ThreadPoolExecutor, as_completed
from typing_extensions import deprecated
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from AutoBioLearn import AutoBioLearn
from decorators import apply_per_grouping, requires_dataset
from helpers import ModelHelper


class AutoBioLearnRegression(AutoBioLearn):
 
    @deprecated("Method will be deprecated, consider using execute_models")
    def run_models(self, models:list[str]=["xgboost"],  times_repeats:int=10, params={}, section:str=None):
        self.execute_models(models, times_repeats,params,section)
    
    def _get_validation(self ,validation: str):
        return ModelHelper.get_validations(validation, "regressor")

    @requires_dataset
    @apply_per_grouping  
    def execute_models(self, models:list[str]=["xgboost"],  times_repeats:int=10, params={}, section:str=None):
                
        models_execution = {}
        if not self.data_processor.dataset.get_has_many_header():
            self._models_executed = []

        unique_models = set(models)
        for model_name in unique_models:
            models_execution[model_name] = ModelHelper.get_model(model_name, "regressor")      

        x = self.data_processor.dataset.get_X(section)
        try:
            y = self.data_processor.dataset.get_Y(section)
        except:
            y = self.data_processor.dataset.get_Y()
        
        params_method = 'quick'

        if "best_params_method" in params:
            params_method = params["best_params_method"]

        if params_method == 'quick':
            model_gen =  RandomizedSearchCV
        else:
            model_gen = GridSearchCV

        params_models = {}

        if "params_models" in params:
            params_models = params["params_models"]
        
        train_size_best_params = 70

        if "best_params_train_size" in params:
            train_size_best_params = params["best_params_train_size"]

        fold_best_params = 5
        if "best_params_n_folds" in params:
            fold_best_params = params["best_params_n_folds"]

        metric_best_params = None
        if "best_params_metrics" in params:
            metric_best_params = params["best_params_metrics"]
            
        def train_test_validation(model_execution):
            executed = []               
            model_name=model_execution[0] 
            model_object, model_params_hidden_verbosity = model_execution[1]

            ix_list_best_params, _ = ModelHelper.initialize_validation(ModelHelper.get_validations("split","regressor"), \
                                                                        0,  \
                                                                        train_size_best_params, \
                                                                        x, y)[0]
            model_params = ModelHelper.get_model_params(model_name,params_models)
            best_params = {}
            if bool(model_params):                 
                model_params = ModelHelper.get_model_params(model_name,params_models)       
                model_instance = model_object()
                            

                model_instance.set_params(**model_params_hidden_verbosity)
                best_params = self._find_best_hyperparams(model_instance, \
                                                        x.iloc[ix_list_best_params], \
                                                        y.iloc[ix_list_best_params], \
                                                        model_params, \
                                                        model_gen, \
                                                        fold_best_params, \
                                                        metric_best_params)
            for current_params in [best_params]:                
                for i in range(times_repeats):                
                        for validation, validation_params in self._validations_execution.items():  
                        
                            ix_list = ModelHelper.initialize_validation(validation_params['validation'], \
                                                                        validation_params['num_folds'],  \
                                                                        validation_params['train_size'], \
                                                                        x, y)

                            for fold, (train_index, test_index) in enumerate(ix_list):
                                x_train = x.iloc[train_index]
                                y_train = y.iloc[train_index]
                                x_test = x.iloc[test_index]
                                y_test = y.iloc[test_index]       

                                model_instance = model_object()
                                merged_params = {**current_params, **model_params_hidden_verbosity}

                                model_instance.set_params(**merged_params)
                                model_instance.fit(x_train, y_train)                                 

                                y_pred = model_instance.predict(x_test)
                                
                                instance = {"time":i,
                                            "validation":validation,
                                            "fold":fold,
                                            "model":model_instance,
                                            "y_pred":y_pred,
                                            "y_test":y_test,
                                            "x_test_index":test_index }
        
                                if section:
                                    instance["section"] = section
                                executed.append(instance)
            return executed         

        with ThreadPoolExecutor() as executor:
            future_to_model = {executor.submit(train_test_validation, models_to_execute): models_to_execute[0] for models_to_execute in models_execution.items()}

            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    models_executed = future.result()
                    for model in models_executed:
                        self._add_model_executed(model["time"],model["validation"], model["fold"], model_name,model["model"],model["y_pred"], model["y_test"],model["x_test_index"], section)
                except Exception as ex:
                   print(ex)
                   
    @apply_per_grouping
    @deprecated("Method will be deprecated, consider using evaluate_models")
    def eval_models(self, metrics: list[str] = ["MSE","RMSE","R2","MAE","MAPE"], section: str = None) -> dict:
        return super().evaluate_models(metrics,section)

    @apply_per_grouping 
    def evaluate_models(self, metrics: list[str] = ["MSE","RMSE","R2","MAE","MAPE"], section: str = None) -> dict:
        return super().evaluate_models(metrics,section)
        
    def _calculate_metrics(self):
        metrics = []
        for row in self._models_executed:
            y_test = row["y_test"]
            y_pred = row["y_pred"]
            if "section" in row:
                metrics.append((row["model_name"], row["section"],row["validation"],row["time"], row["fold"],
                                                        mean_squared_error(y_true= y_test,y_pred= y_pred), \
                                                        root_mean_squared_error(y_true= y_test,y_pred= y_pred), \
                                                        r2_score(y_true= y_test,y_pred= y_pred), \
                                                        median_absolute_error(y_true= y_test,y_pred= y_pred), \
                                                        mean_absolute_percentage_error(y_true= y_test,y_pred= y_pred)))
                
            else:
                metrics.append((row["model_name"], row["validation"],row["time"], row["fold"],
                                                        mean_squared_error(y_true= y_test,y_pred= y_pred), \
                                                        root_mean_squared_error(y_true= y_test,y_pred= y_pred), \
                                                        r2_score(y_true= y_test,y_pred= y_pred), \
                                                        median_absolute_error(y_true= y_test,y_pred= y_pred), \
                                                        mean_absolute_percentage_error(y_true= y_test,y_pred= y_pred)))
        
        if self.data_processor.dataset.get_has_many_header():
             cols_name =["Model", \
                         "Section", \
                        "Validation", \
                        "Time_of_execution", \
                        "Fold", \
                        "MSE", \
                        "RMSE", \
                        "R2","MAE","MAPE"]
        else:
            cols_name =["Model", \
                        "Validation", \
                        "Time_of_execution", \
                        "Fold", \
                        "MSE", \
                        "RMSE", \
                        "R2","MAE","MAPE"]
                
        self._metrics = pd.DataFrame(data = metrics, columns=cols_name)

    @apply_per_grouping     
    def plot_metrics(self, metrics:list[str]=["MSE","RMSE","R2","MAE","MAPE"],rot=90, figsize=(12,6), fontsize=20,section: str = None):
       return super().plot_metrics(metrics = metrics,rot= rot,figsize= figsize, fontsize= fontsize, section= section)
    
