import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt

from helpers.ModelHelper import ModelHelper
class XAIHelper(object):

    @staticmethod
    def default_max_features():
        return 20
    
    @staticmethod
    def get_explainer(model,X):       
        if model["model_name"] in [ModelHelper.const_xgboost(), ModelHelper.const_catboost(), ModelHelper.const_lightboost(), ModelHelper.const_random_forest()]:
            return shap.TreeExplainer(model["model"])      
        else:
            return shap.KernelExplainer(model["model"].predict_proba, X)

    @staticmethod
    def get_consolidate_shap_values(models_explained, model, X, index_to_filter)-> tuple[np.ndarray, pd.DataFrame]:
        models_SHAP_Analisys = list(filter(lambda x: x["model_name"] in model, models_explained))
        
        if index_to_filter is not None:
            shap_values = np.array(models_SHAP_Analisys[0]["shap_values_consolidated"][index_to_filter])
            test_set = models_SHAP_Analisys[0]["x_test_index"][index_to_filter]
        else:
            test_set = models_SHAP_Analisys[0]["x_test_index"]
            shap_values = np.array(models_SHAP_Analisys[0]["shap_values_consolidated"])
                
        for i in range(1,len(models_SHAP_Analisys)):
            if index_to_filter is not None:                
                shap_values = np.concatenate((shap_values,np.array(models_SHAP_Analisys[i]["shap_values_consolidated"][index_to_filter])),axis=0)
                test_set    = np.concatenate((test_set,models_SHAP_Analisys[i]["x_test_index"][index_to_filter]),axis=0)
            else:
                shap_values = np.concatenate((shap_values,np.array(models_SHAP_Analisys[i]["shap_values_consolidated"])),axis=0)
                test_set    = np.concatenate((test_set,models_SHAP_Analisys[i]["x_test_index"]),axis=0)
        
       
        #shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=0)
    
        X_test = pd.DataFrame(X.iloc[test_set],columns=X.columns)

        return shap_values, X_test
    
    @staticmethod
    def get_consolidate_shap_values_lightboost(models_explained, model, X, class_index,index_to_filter)-> tuple[np.ndarray, pd.DataFrame]:
        models_SHAP_Analisys = list(filter(lambda x: x["model_name"] in model, models_explained))
        
        if index_to_filter is not None:
            shap_values = np.array(models_SHAP_Analisys[0]["shap_values_consolidated"][class_index][index_to_filter])
            test_set = models_SHAP_Analisys[0]["x_test_index"][index_to_filter]
        else:
            test_set = models_SHAP_Analisys[0]["x_test_index"]
            shap_values = np.array(models_SHAP_Analisys[0]["shap_values_consolidated"][class_index])
                
        for i in range(1,len(models_SHAP_Analisys)):
            if index_to_filter is not None:                
                shap_values = np.concatenate((shap_values,np.array(models_SHAP_Analisys[i]["shap_values_consolidated"][class_index][index_to_filter])),axis=0)
                test_set    = np.concatenate((test_set,models_SHAP_Analisys[i]["x_test_index"][index_to_filter]),axis=0)
            else:
                shap_values = np.concatenate((shap_values,np.array(models_SHAP_Analisys[i]["shap_values_consolidated"][class_index])),axis=0)
                test_set    = np.concatenate((test_set,models_SHAP_Analisys[i]["x_test_index"]),axis=0)
        
       
        #shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=0)
    
        X_test = pd.DataFrame(X.iloc[test_set],columns=X.columns)

        return shap_values, X_test

    @staticmethod
    def get_chart_type_local(graph_type, expected_value,shap_values,X,params, show_all_features = True):
            max_features = XAIHelper.default_max_features()

            if(show_all_features):
                max_features = X.shape[0]

            if "force" in graph_type:
                shap.force_plot(expected_value,shap_values,X, matplotlib=True)     
            elif "waterfall" in graph_type:
                shap.plots._waterfall.waterfall_legacy(expected_value,
                                                        shap_values,
                                                        features=X)
                #shap.waterfall_plot(shap_values)
            elif "violin" in graph_type:
                shap.summary_plot(shap_values, X, title="SHAP summary plot",plot_type="violin", show=False,max_display=max_features)
            elif "bar" in graph_type:
                shap.summary_plot(shap_values, X, title="SHAP summary plot",plot_type="bar", show=False,max_display=max_features)
            plt.show()

    @staticmethod
    def get_chart_type_global(graph_type,shap_values,X,params, show_all_features = True):
            max_features = XAIHelper.default_max_features()

            if(show_all_features):
                max_features = X.shape[1]

            if "summary" in graph_type:
                shap.summary_plot(shap_values, X, title="SHAP summary plot", show=False,max_display=max_features)       
            elif "dependence" in graph_type:
                col_axel_y = None
           
                graph_params = params.get("graph_params")

                if graph_params != None and graph_params.get("dependence") != None:
                    dependence_params =  graph_params.get("dependence")                  
                    col_axel_y = dependence_params.get("col_axel_y")
                
                if col_axel_y is None:
                    col_axel_y = X.columns               
              

                for col in col_axel_y:
                    shap.dependence_plot(col, shap_values, X)

            elif "violin" in graph_type:
                shap.summary_plot(shap_values, X, title="SHAP summary plot",plot_type="violin", show=False,max_display=max_features) 
            elif "bar" in graph_type:
                shap.summary_plot(shap_values, X, title="SHAP summary plot",plot_type="bar", show=False,max_display=max_features)
            
            plt.show()      