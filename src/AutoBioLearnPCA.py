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
        y = self.data_processor.dataset.get_Y(section)
        
        self.pca = PCA(n_components=n_components)
        self.scores = self.pca.fit_transform(df)
        
        components_cols = [f'PC{i}' for i in range(1, n_components+1)]
        self.coordinates = pd.DataFrame(self.scores,
                                        index=df.index, 
                                        columns=components_cols)
        self.coordinates['class'] = y 

          
    @requires_dataset
    def __kmo(self):
        """
        Kaiser–Meyer–Olkin (KMO)
        MO should be one in the ideal case. 
        High KMO values indicate a PCA with few errors, overall.
        If KMO is more than 0.5, PCA could be used
        """
        _,kmo_model=calculate_kmo(self.data_processor.dataset.get_X())
        self.kmo = kmo_model

    
    @requires_dataset
    def __bartlett(self):
        """
        The null hypothesis is that the intercorrelation matrix comes from 
        a noncollinear populaton or simply that there is no collinearity 
        between the variables, which would render PCA impossible as it depends
        on the construction of a linear combination of the variables.
        """
        chi,p =calculate_bartlett_sphericity(self.data_processor.dataset.get_X())
        self.bartlett = {'p-val':p, 'chi-squared':chi}
             
    
    @requires_dataset
    def __kaiser(self, max_dim=10):
        """
        The Kaiser criterion is a method for determining how many principal 
        components (PCs) to retain in a principal components analysis (PCA).
        It counts how many eigenvalues are >1.
        """
        eigenvalues = self.pca.explained_variance_
        self.kaiser = len(np.where(eigenvalues > 1)[0])


    @requires_dataset
    def cumulative_var(self, thresh, save=True):

        var_ratio = np.cumsum(self.pca.explained_variance_ratio_)
        fig, ax = plt.subplots()
        sns.lineplot(var_ratio, ax=ax)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.hlines(y=thresh, xmin=0, xmax=len(var_ratio), color='r')
        plt.show()
        if save == True:
            fig.savefig('cumulative_variance.png', format='png')


    @requires_dataset
    def scree(self, save=True):

        var = self.pca.explained_variance_
        fig, ax = plt.subplots()
        sns.lineplot(var, ax=ax)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Explained variance')
        ax.set_title('Scree plot')
        plt.show()
        if save == True:
            fig.savefig('scree.png', format='png')

    
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
                
        # PCA plot
        PC1_var= self.pca.explained_variance_ratio_[0]
        PC2_var= self.pca.explained_variance_ratio_[1]
        
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7), dpi = 600)
        sns.scatterplot(data=self.coordinates,
                        x='PC1',
                        y='PC2',
                        hue='class',
                        palette=cmap,
                        ax=axes,
                        legend='brief')
        plt.xlabel(f'PC1 (explained variance: {str(PC1_var * 100)[:6]}%)')
        plt.ylabel(f'PC2 (explained variance: {str(PC2_var * 100)[:6]}%)')
        
        # Save the figure
        if save == True:
            fig.savefig('PCA.png', format='png')
        plt.show()
        plt.cla()

