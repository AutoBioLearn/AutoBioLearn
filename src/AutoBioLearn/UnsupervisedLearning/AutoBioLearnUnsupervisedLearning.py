
from abc import ABC, abstractmethod
from AutoBioLearn.AutoBioLearnBase import AutoBioLearnBase

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

class AutoBioLearnUnsupervisedLearning(AutoBioLearnBase,ABC):

    def __init__(self):
        super().__init__()
        self._models_executed = {} 

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


    def _metric_options(self, key):
        options = {'silhouette_euclidean':(self.silhouette, {'metric':'euclidean'}),
                   'silhouette_cosine':   (self.silhouette, {'metric':'cosine'   }),
                   'l1':                  (self.silhouette, {'metric':'l1'       }),
                   'l2':                  (self.silhouette, {'metric':'l2'       }),
                   'manhattan':           (self.silhouette, {'metric':'manhattan'}),
                   'calinski_harabasz':   (self.calinski_harabasz,              {}),
                   'davies_bouldin':      (self.davies_bouldin,                 {})}
        return options[key]


    @abstractmethod
    def run(self):
        pass


    @abstractmethod
    def plot(self):
        pass