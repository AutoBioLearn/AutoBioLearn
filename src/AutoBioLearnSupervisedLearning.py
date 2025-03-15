from abc import ABC, abstractmethod
from AutoBioLearn import AutoBioLearn

class AutoBioLearnSupervisedLearning(AutoBioLearn,ABC):

    def __init__(self) -> None:
        super().__init__()
        self._models_executed = []
        self._validations_execution = {}
        validation_object =self._get_validation("split")
        self._validations_execution["split"] = {
            'validation': validation_object,
            'num_folds': 0,
            'train_size': 70
        }  

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

    def _add_model_executed(self ,time: int,validation: str, fold: int,                                                         
                            model_name: str, model,y_pred, y_prob, y_test, x_test_index, section= None):
        
        instance = {"time":time,
                    "validation":validation,
                    "fold":fold,                                                        
                    "model_name":model_name,
                    "model":model,
                    "y_pred":y_pred,
                    'y_prob':y_prob,
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