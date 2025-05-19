
from abc import ABC, abstractmethod
from AutoBioLearn import AutoBioLearn

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

class AutoBioLearnUnsupervisedLearning(AutoBioLearn,ABC):
    
    def silhouette(self,
                   y_pred,
                   section:str=None):
        X = self.data_processor.dataset.get_X(section)
        return silhouette_score(X, y_pred)


    def calinski_harabasz(self,
                          y_pred,
                          section:str=None):
        X = self.data_processor.dataset.get_X(section)
        return calinski_harabasz_score(X, y_pred)

    
    def davies_bouldin(self,
                       y_pred,
                       section:str=None):
        X = self.data_processor.dataset.get_X(section)
        return davies_bouldin_score(X, y_pred)
    
    # def _find_best_hyperparams(self,
    #                            metric=silhouette,
    #                            methods:list=['ward', 'complete', 'average', 'single'],
    #                            metric:list=[],
    #                            max_clusters:int=10):

    #     range_n_clusters = range(2, max_clusters)
    #     best_n = 2

    #     best_config = None
        
    #     for method in methods:
    #         for metric in ['euclidean', 'cityblock']:
    #             for k in range(2, 11):
    #                 try:
    #                     # 'ward' only works with 'euclidean'
    #                     if method == 'ward' and metric != 'euclidean':
    #                         continue
    #     for n_clusters in range_n_clusters:
    #         # Run with the set n of clusters. Could I make it with other types of metrics too?
    #         current = # calculate metric. Make it in a way that it works with davies too.
    #         if current > best_score:
    #             best_n = n_clusters
    #             best_score = current
        
    #     self._optimal_n_clusters = best_score