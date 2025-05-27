import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import pandas as pd
import numpy as np

import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

from AutoBioLearnUnsupervisedLearning import AutoBioLearnUnsupervisedLearning
from decorators import requires_dataset
from helpers import ModelHelper

class AutoBioLearnHierarchical(AutoBioLearnUnsupervisedLearning):
    
    def __init__(self) -> None:
        super().__init__()
    
    
    @requires_dataset
    def execute_models(self,
                      method:str='average',
                      metric:str='euclidean',
                      n_clusters:int=None,
                      criterion:str='inconsistent',
                      optimize:bool=False,
                      section:str=None):
            """
            method = 'single', 'average', 'complete', 'ward', 'centroid', etc
            metric = 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 
                     'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 
                     'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis',
                     'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 
                     'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
                     'yule'.
            """
            # Get data
            X = self.data_processor.dataset.get_X(section)
            self._Hclustering = sch.linkage(X,
                                            method = method,
                                            metric = metric)
                
            if n_clusters == None:
                try:
                    t = self._optimal_n_clusters  # what?
                except:
                    t = '3'
            else:
                t = n_clusters
            
            self.predicted_cluster = fcluster(self._Hclustering,
                                              t=t,
                                              criterion=criterion)
            
            # For this becoming like you did below for partitioning.
            # A lot of work.
            # 1- Mind of sections
            # 2 - You'd actually have a grid of metrics and methods.
            # What is on the cells of this grid?
            # It should store the number of clusters.
            # And the sch.linkage object?
            

    def cophenetic_corr():
        pass


    def _calculate_metrics(self):
        print('Not really implemented yet') 
        # Just run the cophenetic correlation!

    def evaluate_models(self):
        print('Not really implemented yet')   
   
    
    @requires_dataset
    def heatmap(self,
                method:str='average',
                metric:str='euclidean',
                cmap:str='plasma_r',
                section:str=None,
                save:bool=True):
        """
        method = 'single', 'average', 'complete', 'ward', 'centroid', etc
        metric = 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 
                 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 
                 'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis',
                 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 
                 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
                 'yule'.
        """
        
        # Get data
        X = self.data_processor.dataset.get_X(section)
        try:
            y = self.data_processor.dataset.get_Y(section)
        except:
            y = self.data_processor.dataset.get_Y()
        
        # Add colour to the class
        colours = sns.color_palette("husl", len(y.unique())).as_hex()
        colours = dict(zip(y.unique(), colours))
        group = y.replace(colours)

        # Plot heatmap
        fig = sns.clustermap(X,
                             row_cluster=False,
                             method=method,
                             metric=metric,
                             z_score=None,
                             standard_scale=None,
                             figsize=(8, 12),
                             row_colors=group,
                             cmap= cmap)
        
        plt.title(f'Dendrogram - {method}', fontsize=16)
        plt.ylabel(f'{metric}', fontsize=16)
        
        # Add legend to class
        handles = [mpatches.Patch(color=color, label=label) for label, color in colours.items()]
        plt.legend(handles=handles, bbox_to_anchor=(1.2, 1), loc='lower left')
        
        # Save it
        if save == True:
            fig.savefig(f'heatmap_{metric}_{method}.png')
            
        plt.show()
        plt.cla()
    
    @requires_dataset
    def dendogram(self,
                  method:str='average',
                  metric:str='euclidean',
                  thresh:int=3,
                  section:str=None,
                  save:bool=True):
        """
        method = 'single', 'average', 'complete', 'ward', 'centroid', etc
        metric = 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 
                 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 
                 'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis',
                 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 
                 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
                 'yule'.
        """

        # Plot
        fig, axis = plt.subplots(figsize=(8,12))
        
        sch.dendrogram(self._Hclustering,
                       labels = self.data_processor.dataset.get_X.index,
                       ax=axis,
                       orientation='left',
                       color_threshold=thresh)
        plt.title(f'Dendrogram - {method}', fontsize=16)
        plt.ylabel(f'{metric}', fontsize=16)
        
        # Save it
        if save == True:
            fig.savefig(f'dendogram_{metric}_{method}.png')
            
        plt.show()
        plt.cla()
  

###############################################################################

class AutoBioLearnPartitional(AutoBioLearnUnsupervisedLearning):
    
    def __init__(self) -> None:
        super().__init__()
    
    def _metric_options(self, key):
        options = {'silhouette_euclidean':(super().silhouette, {'metric':'euclidean'}),
                   'silhouette_cosine':   (super().silhouette, {'metric':'cosine'   }),
                   'l1':                  (super().silhouette, {'metric':'l1'       }),
                   'l2':                  (super().silhouette, {'metric':'l2'       }),
                   'manhattan':           (super().silhouette, {'metric':'manhattan'}),
                   'calinski_harabasz':   (super().calinski_harabasz,              {}),
                   'davies_bouldin':      (super().davies_bouldin,                 {})}
        return options[key]

    def run(self,
            model,
            nclusters,
            section:str=None,
            metrics:list[str]=['silhouette_euclidean',
                               'silhouette_cosine',
                               'l1',
                               'l2',
                               'manhattan',
                               'calinski_harabasz',
                               'davies_bouldin'],
            print_met=False):

            try:
                X = self.data_processor.dataset.get_X(section)
            except KeyError:
                X = self.data_processor.dataset.get_X()

            model = ModelHelper.get_model(model, "clustering")
            model = model(n_clusters=nclusters)
            try:
                model.fit(X)
                yhat = model.predict(X)
            except AttributeError:
                yhat = model.fit_predict(X)
            
            self._current_model = {'results' : yhat,
                                   'params'  : (section, model, nclusters)}
            
            if print_met == True:
                metrics = {key : self._metric_options(key) for key in metrics}
                for met, (function, kargs) in metrics.items():
                    print(f'{met} = {function(yhat, **kargs)}')
                    

    def execute_models(self,
                       models:list[str]=['kmeans', 'spectral', 'birch'],
                       cluster_range:tuple=(2,5),
                       section: str = None):
        
        models_execution = {}
        unique_models = set(models)
      
        for name in unique_models:
            models_execution[name] = {}
            for i in range(cluster_range[0], cluster_range[1]+1):
                self.run(name, i, section)
                models_execution[name][i] = self._current_model['results']

        section_name = section if section != None else 'all variables'

        if self._models_executed == []:
            self._models_executed = {(section_name, key): vals for key, vals in models_execution.items()}
        else:
            for key, vals in models_execution.items(): 
                self._models_executed[(section, key)] = vals
    

    def _calculate_metrics(self,
                           metrics:list[str]=['silhouette_euclidean', 
                                              'calinski_harabasz',
                                              'davies_bouldin']):
        
        yhats = self._models_executed
        scores = {}
    
        metrics = {key : self._metric_options(key) for key in metrics}

        for (section, algorithm), clusters in yhats.items():
            for met, (function, kargs) in metrics.items():
                results = {k : function(v, **kargs) for k, v in clusters.items()}
                scores[(section, algorithm, met)] = results
        
        metrics = pd.DataFrame(scores)
        metrics.index = metrics.index.rename('Number of clusters')
        metrics.columns = metrics.columns.rename(['Section',
                                                  'Method',
                                                  'Metric'])
        self.metrics = metrics


    def evaluate_models(self,
                        criterion:str='silhouette_euclidean',
                        metrics:list[str]=['silhouette_euclidean',
                                           'calinski_harabasz',
                                           'davies_bouldin'],
                        section: str = 'all variables',
                        figure=True):
        
        self._calculate_metrics(metrics)

        print(f'Models will be evaluated by {criterion} \n')
        subset = self.metrics.xs(criterion, level=2, axis=1)
        subset = subset.xs(section, level=0, axis=1)

        if figure == True:
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(subset, ax=ax)
            plt.title(f'Dendrogram - {criterion}', fontsize=16)
            fig.savefig(f'{criterion}_clusters_methods.png')
        
        print(subset)        
        stack = subset.stack()
        for m, func in {'Max': (lambda x: x.idxmax()),
                        'Min': (lambda x: x.idxmin())}.items():
            a = func(stack)
            print(f"{m} value:{a[0]} clusters, {a[1]}")
            if (m == 'Max' and criterion != 'davies_bouldin') \
                or (m == 'Min' and criterion == 'davies_bouldin'):
                self._best_params = {'model':a[1],
                                     'nclusters':a[0],
                                     'section':section}
                self.run(**self._best_params)


    def plot(self,
             x_axis,
             y_axis):

        yhat, title = self._current_model.values()
        section = title[0]
        
        try:
            X = self.data_processor.dataset.get_X(section)
        except KeyError:
            X = self.data_processor.dataset.get_X()
        
        if x_axis not in X.columns or y_axis not in X.columns:
            raise ValueError(f"{x_axis} and/or {y_axis} not in dataset.")

        try:
            X = self.data_processor.dataset.get_X(section)
        except KeyError:
            X = self.data_processor.dataset.get_X()

        clusters = set(yhat)
        fig, ax = plt.subplots()
        
        # create scatter plot for samples from each cluster
        for cluster in clusters:
            row_ix = np.where(yhat == cluster)[0]
            ax.scatter(
                X.iloc[row_ix][x_axis],
                X.iloc[row_ix][y_axis],
                label=f'Cluster {cluster}'
            )
        
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)

        plt.title(f'Section: {title[0]}, {title[1]}, {title[2]} clusters')
        fig.savefig(f'sec_{title[0]}-{title[1]}-{title[2]}-{x_axis}-{y_axis}.png')

        plt.cla()

