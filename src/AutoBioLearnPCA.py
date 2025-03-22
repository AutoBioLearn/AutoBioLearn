import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
from AutoBioLearnUnsupervisedLearning import AutoBioLearnUnsupervisedLearning
from decorators import requires_dataset

class AutoBioLearnPCA(AutoBioLearnUnsupervisedLearning):
    
    def __init__(self) -> None:
        super().__init__()

    @requires_dataset
    def PCA(self,
            n_components:int=2,
            section:str=None,): # See if it is a class method or whatever
        #TODO
        # Get data
        df = self.data_processor.dataset
        
        pca = PCA(n_components=n_components)
        pca = pca.fit(df)  # Set as class attribute
        coordinates = pca.transform(df)
        coordinates = pd.DataFrame(coordinates,  # Set as class attribute
                                   index=df.index,
                                   columns=['PC1', 'PC2'])
        
    @requires_dataset
    def bartlett():
        #TODO
        pass
    
    @requires_dataset
    def eigenvalues():
        #TODO
        pass

    @requires_dataset
    def variance():
        #TODO
        pass

    @requires_dataset
    def loading_table():
        #TODO
        pass
    
    @requires_dataset
    def loading_plot():
        #TODO
        pass

    @requires_dataset
    def variance_plot():
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

