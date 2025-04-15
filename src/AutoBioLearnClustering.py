import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy
from AutoBioLearnUnsupervisedLearning import AutoBioLearnUnsupervisedLearning
from decorators import requires_dataset

class AutoBioLearnHClustering(AutoBioLearnUnsupervisedLearning):
    
    def __init__(self) -> None:
        super().__init__()
            
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
        
        # Get data
        X = self.data_processor.dataset.get_X(section)
        
        # Plot
        fig, axis = plt.subplots(figsize=(8,12))
        dend = sch.linkage(X,
                           method = method,
                           metric = metric)
        sch.dendrogram(dend,
                       labels = X.index,
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
        

    def execute_models(self):
        print('Not really implemented yet')   


    def evaluate_models(self):
        print('Not really implemented yet')   


    def _calculate_metrics(self):
        print('Not really implemented yet')   
