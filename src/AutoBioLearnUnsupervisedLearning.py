
from abc import ABC, abstractmethod
from AutoBioLearn import AutoBioLearn

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

class AutoBioLearnUnsupervisedLearning(AutoBioLearn,ABC):
    
    def silhouette(self,
                   y_pred,
                   metric:str='euclidean',
                   section:str=None):
        X = self.data_processor.dataset.get_X(section)
        return silhouette_score(X, y_pred, metric=metric)


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
    