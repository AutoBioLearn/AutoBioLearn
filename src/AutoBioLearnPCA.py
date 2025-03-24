import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import bartlett
import pandas as pd
from AutoBioLearnUnsupervisedLearning import AutoBioLearnUnsupervisedLearning
from decorators import requires_dataset

class AutoBioLearnPCA(AutoBioLearnUnsupervisedLearning):
    
    def __init__(self) -> None:
        super().__init__()

    @requires_dataset
    def execute_models(self,
                       n_components:int=2,
                       section:str=None):
        # Get data
        df = self.data_processor.dataset
        
        self.pca = PCA(n_components=n_components)
        self.pca = self.pca.fit_transform(df)
        
        components_cols = [f'PC{i}' for i in range(1, n_components+1)]
        self.coordinates = pd.DataFrame(self.pca,
                                        index=df.index,
                                        columns=components_cols)

    @requires_dataset
    def __kmo(self):
        #TODO
        pass

    
    @requires_dataset
    def __bartlett(self):

        df = self.data_processor.dataset
        self.bartlett = bartlett(*[df[col] for col in df.columns])
        
    
    @requires_dataset
    def __eigenvalues(self):
        #TODO
        pass

    @requires_dataset
    def __variance(self):
        #TODO
        pass
    
    @requires_dataset
    def _calculate_metrics(self):
        pass
    
    def evaluate_models(self):
        pass

    @requires_dataset
    def loading_table(self):
        #TODO
        pass
    
    @requires_dataset
    def loading_plot(self):
        #TODO
        pass

    @requires_dataset
    def variance_plot(self):
        #TODO
        pass
    
    @requires_dataset
    def PCA_plot(self,
                 cmap:str='muted',
                 save:bool=True):
        
        # Print statistics #TODO
        
        # PCA plot
        PC1_var= self._pca.explained_variance_ratio_[0]
        PC2_var= self._pca.explained_variance_ratio_[1]
        
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7), dpi = 600)
        sns.scatterplot(data=self._coordinates,
                        x='PC1',
                        y='PC2',
                        hue=self.__target,
                        palette=cmap,
                        legend=False,
                        ax=axes)
        plt.xlabel(f'PC1 (explained variance: {str(PC1_var * 100)[:6]}%)')
        plt.ylabel(f'PC2 (explained variance: {str(PC2_var * 100)[:6]}%)')
        
        # Save the figure
        if save == True:
            fig.savefig('PCA.png', format='png')
            
        plt.show()
        plt.cla()

