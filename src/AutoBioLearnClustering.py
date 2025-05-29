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


    def _cophenetic_corr(self,
                         Z,
                         section:str=None,
                         print_res=True):
        try:
            X = self.data_processor.dataset.get_X(section)
        except KeyError:
            X = self.data_processor.dataset.get_X()
        
        c, _ = sch.cophenet(Z, scipy.spatial.distance.pdist(X))
        if print_res == True:
            if c >= 0.75:
                print(f'Cophenetic correlation is {c}: good fit')
            elif c >= 0.5:
                print(f'Cophenetic correlation is {c}: moderate fit')
            else:
                print(f'Cophenetic correlation is {c}: poor fit')
        return c


    @requires_dataset
    def run(self,
            method:str='average',
            n_clusters:int=3,
            section:str=None,
            metric:str='euclidean',
            print_met=False):
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
            try:
                X = self.data_processor.dataset.get_X(section)
            except KeyError:
                X = self.data_processor.dataset.get_X()

            model = sch.linkage(X,
                                method = method,
                                metric = metric)

            yhat = fcluster(model,
                            t=n_clusters,
                            criterion='maxclust')
            
            self._current_model = {'results' : yhat,
                                   'params'  : (section, model, metric, n_clusters),
                                   'object'  : model}

            if print_met == True:
                metrics = ['silhouette_euclidean',
                           'silhouette_cosine',
                           'calinski_harabasz',
                           'davies_bouldin']
                metrics = {key : self._metric_options(key) for key in metrics}
                for met, (function, kargs) in metrics.items():
                    print(f'{met} = {function(yhat, **kargs)}')
                self._cophenetic_corr(model, X)


    def execute_models(self,
                       method:list[str]=['average', 'single', 'complete'],
                       cluster_range:tuple=(2, 5),
                       metric:str=['euclidean', 'braycurtis', 'canberra', 'correlation'],
                       section:str=None):

        section_name = section if section is not None else 'all variables'
    
        for m in method:
            for d in metric:
                col_key = (section_name, m, d)
                
                if col_key not in self._models_executed:
                    self._models_executed[col_key] = {}
                
                for k in range(cluster_range[0], cluster_range[1] + 1):
                    self.run(method=m, n_clusters=k, section=section, metric=d)
                    self._models_executed[col_key][k] = self._current_model


    def _calculate_metrics(self,
                           metrics:list[str]=['cophenetic',
                                              'silhouette_euclidean', 
                                              'calinski_harabasz',
                                              'davies_bouldin']):
        results = self._models_executed
        scores = {}

        metrics_yhat = {k : self._metric_options(k) for k in metrics if k != 'cophenetic'}

        for (s, m, d), clusters in results.items():
            for met, (function, kargs) in metrics_yhat.items():
                l = {}
                for k, v in clusters.items():
                    if len(set(v['results'])) <= 1:
                        l[k] = np.nan
                        print(f"{s}, {m}, {d} with only 1 cluster. {met} set to NaN.")
                    else:
                        l[k] = function(v['results'], **kargs)
                scores[(s, m, d, met)] = l

        if 'cophenetic' in metrics:
            l = {
                k : self._cophenetic_corr(v['object'], s, print_res=False)
                for k, v in clusters.items()}
            scores[(s, m, d, 'cophenetic')] = l

        metrics = pd.DataFrame(scores)
        metrics.index = metrics.index.rename('Number of clusters')
        metrics.columns = metrics.columns.rename(['Section',
                                                  'Method',
                                                  'Clustering metric',
                                                  'Evaluation metric'])
        self.metrics = metrics


    def evaluate_models(self,
                        criterion:str='cophenetic',
                        metrics:list[str]=['cophenetic',
                                           'silhouette_euclidean', 
                                           'calinski_harabasz',
                                           'davies_bouldin'],
                        section: str = 'all variables'):

        self._calculate_metrics(metrics)

        print(f'Models will be evaluated by {criterion} \n')
        subset = self.metrics.xs(criterion, level=3, axis=1)
        subset = subset.xs(section, level=0, axis=1)

        print(subset)
        stack = subset.stack()
        for m, func in {'Max': (lambda x: x.idxmax()),
                        'Min': (lambda x: x.idxmin())}.items():
            a = func(stack)
            print(f"{m} value:{a[0][0]} clusters, metric {a[0][1]}, \
                  {a.index[0]} linkage")
            if (m == 'Max' and criterion != 'davies_bouldin') \
                or (m == 'Min' and criterion == 'davies_bouldin'):
                self._best_params = {'metric':a[0][1],
                                     'method': a.index[0],
                                     'n_clusters':a[0][0],
                                     'section':section}
                print(self._best_params)
                self.run(**self._best_params)
   
    
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
        
    def plot(self):
        print('nope')
  

###############################################################################

class AutoBioLearnPartitional(AutoBioLearnUnsupervisedLearning):
    
    def __init__(self) -> None:
        super().__init__()

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
                self._models_executed[(section_name, key)] = vals
    

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
        
        if subset.isnull().all().all():
            print('No available results for this criterion and section')
            return

        if figure == True:
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(subset, ax=ax)
            plt.title(f'{criterion}', fontsize=16)
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

        plt.title(f'Section: {title[0]}, {title[1]}')
        fig.savefig(f'sec_{title[0]}-{title[1]}-{title[2]}-{x_axis}-{y_axis}.png')

        plt.cla()

