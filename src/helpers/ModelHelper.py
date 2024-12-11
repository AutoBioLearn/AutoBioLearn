from catboost import CatBoostClassifier,CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold,LeaveOneOut, StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier,XGBRegressor


class ModelHelper(object):

    @staticmethod
    def const_xgboost()-> str:
        return "xgboost"
    
    @staticmethod
    def const_catboost()-> str: 
        return "catboost"
    
    @staticmethod
    def const_lightboost()-> str: 
        return "lightboost"
    
    @staticmethod
    def const_xgboost()-> str: 
        return "xgboost"
    
    @staticmethod
    def const_random_forest()-> str: 
        return "random_forest"
    
    @staticmethod
    def const_logistic_regression()-> str: 
        return "logistic_regression"
    
    @staticmethod
    def const_svm()-> str: 
        return "svm"
    
    @staticmethod  
    def get_model(model, model_type="classifier"):
        if model is None or model_type is None:
            raise AttributeError()
        
        classifiers ={  ModelHelper.const_xgboost():  (XGBClassifier, {}),
                        ModelHelper.const_catboost(): (CatBoostClassifier,{"allow_writing_files":False, "verbose": False}),
                        ModelHelper.const_lightboost():(LGBMClassifier,{"verbosity":-1}),
                        ModelHelper.const_random_forest(): (RandomForestClassifier,{}),
                        ModelHelper.const_logistic_regression():(LogisticRegression,{}),
                        ModelHelper.const_svm():(SVC,{})
                    }
        
        regressors ={
                        ModelHelper.const_xgboost():  (XGBRegressor,{}),
                        ModelHelper.const_catboost(): (CatBoostRegressor,{"allow_writing_files":False, "verbose": False}),
                        ModelHelper.const_lightboost():(LGBMRegressor,{"verbosity":-1}),
                        ModelHelper.const_svm():(SVR,{}),
                        ModelHelper.const_random_forest(): (RandomForestRegressor,{}),
                    }
        
        if model_type.lower() == 'classifier':
            return classifiers[model.lower()]
        elif model_type.lower() == 'regressor':
            return regressors[model.lower()]
        

    @staticmethod  
    def get_validations(validation, model_type ="classifier"):
        validations_classification={
            "kfold": KFold,
            "stratified_kfold": StratifiedKFold,
            "leave_one_out": LeaveOneOut,
            "split" : StratifiedShuffleSplit
        }

        validations_regression ={
            "kfold": KFold,            
            "leave_one_out": LeaveOneOut,
            "split" : ShuffleSplit
        }
        
        if model_type.lower() == 'classifier':
            return validations_classification[validation.lower()]
        elif model_type.lower() == 'regressor':
            return validations_regression[validation.lower()]  

    @staticmethod  
    def initialize_validation(validation_object, num_folds: int,train_size:float, X, y):
        validation_name = str(validation_object).lower()
        if 'kfold' in validation_name:
            return list(validation_object(n_splits= num_folds,shuffle=True).split(X,y))
        elif "leaveoneout" in validation_name:
            return list(validation_object().split(X,y))
        else:
            return list([next(validation_object(train_size=(train_size/100)).split(X,y))])
           
    
    @staticmethod  
    def get_model_params(model, params)-> dict:
        model_param ={}
        
        for k, v in params.items():
            if model in k:
                model_param[k.replace(model+"_","")]= v

        return model_param
