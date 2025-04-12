import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

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
        df = self.data_processor.dataset.get_X(section)
        #y = self.data_processor.dataset.get_Y(section)
        
        self.pca = PCA(n_components=n_components)
        self.scores = self.pca.fit_transform(df)
        
        components_cols = [f'PC{i}' for i in range(1, n_components+1)]
        self.coordinates = pd.DataFrame(self.scores,
                                        index=df.index, # TODO colocar target
                                        columns=components_cols)

    @requires_dataset
    def __kmo(self):
        _,kmo_model=calculate_kmo(self.data_processor.dataset.get_X())
        self.kmo = kmo_model

    
    @requires_dataset
    def __bartlett(self):
        chi,p =calculate_bartlett_sphericity(self.data_processor.dataset.get_X())
        self.bartlett = {'p-val':p, 'chi-squared':chi}
             
    
    @requires_dataset
    def __kaiser(self, max_dim=10):
        self.execute_models(n_components=max_dim)
        
        eigenvalues = self.pca.explained_variance_
        self.kaiser = len(np.where(eigenvalues > 1)[0])

    @requires_dataset
    def __variance_exp(self, max_dim, thresh):
        self.execute_models(n_components=max_dim)
        var_ratio = np.cumsum(self.pca.explained_variance_ratio_)
        n = len(np.where(var_ratio < thresh)[0]) + 1
        var = var_ratio[n]
        self.var_explained = {'n_components': n, 'explained variance': var}
        
        fig, ax = plt.subplots()
        sns.lineplot(var_ratio, ax=ax)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.hlines(y=thresh, color='r')
        plt.show()


    @requires_dataset
    def __scree(self):
        #TODO
        pass
    
    @requires_dataset
    def _calculate_metrics(self):
        self.__kmo()
        print(self.kmo)
        
        self.__bartlett()
        print(self.bartlett)
    
    def evaluate_models(self):
                
        # print(f'Chi-squared: {chi}')
        # print('P-value: {p}')
        
        # if p > 0.05:
        #     print('P-value above 0.05, we advise not employ a PCA')
        # else:
        #     print('P-value below 0.05, you may employ a PCA')
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
        PC1_var= self.pca.explained_variance_ratio_[0]
        PC2_var= self.pca.explained_variance_ratio_[1]
        
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7), dpi = 600)
        sns.scatterplot(data=self.coordinates,
                        x='PC1',
                        y='PC2',
                        #hue=self.data_processor.dataset.get_target_name(),
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

