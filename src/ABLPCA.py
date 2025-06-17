import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA as sklPCA
import pandas as pd
import numpy as np

from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

from ABLUnsupervised import Unsupervised
from decorators import requires_dataset

class PCA(Unsupervised):
    
    def __init__(self) -> None:
        super().__init__()

    @requires_dataset
    def run(self,
            n_components:int=2,
            section:str=None):
        # Get data
        df = self.data_processor.dataset.get_X(section)
        y = self.data_processor.dataset.get_Y(section, recode=False)
        
        if n_components is None:
            self.pca = sklPCA()
            n_components = min(df.shape)
        else:
            self.pca = sklPCA(n_components=n_components)
        self.scores = self.pca.fit_transform(df)
        
        components_cols = [f'PC{i}' for i in range(1, n_components+1)]
        self.coordinates = pd.DataFrame(self.scores,
                                        index=df.index, 
                                        columns=components_cols)
        self.coordinates['class'] = y


    def execute_models():
        print('Not applicable to PCA')


    @requires_dataset
    def _kmo(self,
             section:str=None):
        """
        Kaiser–Meyer–Olkin (KMO)
        MO should be one in the ideal case. 
        High KMO values indicate a PCA with few errors, overall.
        If KMO is more than 0.5, PCA could be used
        """
        _,kmo_model=calculate_kmo(self.data_processor.dataset.get_X(section))
        self.kmo = kmo_model

    
    @requires_dataset
    def _bartlett(self,
                  section:str=None):
        """
        The null hypothesis is that the intercorrelation matrix comes from 
        a noncollinear populaton or simply that there is no collinearity 
        between the variables, which would render PCA impossible as it depends
        on the construction of a linear combination of the variables.
        """
        chi,p =calculate_bartlett_sphericity(self.data_processor.dataset.get_X(section))
        self.bartlett = {'p-val':p, 'chi-squared':chi}
             
    
    @requires_dataset
    def _kaiser(self,
                section:str=None):
        """
        The Kaiser criterion is a method for determining how many principal 
        components (PCs) to retain in a principal components analysis (PCA).
        It counts how many eigenvalues are >1.
        """
        df = self.data_processor.dataset.get_X(section)
        corr_matrix = np.corrcoef(df.T)

        eigvals, _ = np.linalg.eigh(corr_matrix)
        self.kaiser = len(np.where(eigvals > 1)[0])


    @requires_dataset
    def cumulative_var(self, thresh, save=True):

        var_ratio = np.cumsum(self.pca.explained_variance_ratio_)
        fig, ax = plt.subplots()
        sns.lineplot(x=range(1, len(var_ratio)+1),
                     y=var_ratio,
                     ax=ax)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.hlines(y=thresh, xmin=0, xmax=len(var_ratio), color='r')
        plt.show()
        if save == True:
            fig.savefig('cumulative_variance.png', format='png')
        
        i = 0
        val = 0
        while val < thresh:
            i += 1 
            val = var_ratio[i]

        if hasattr(self, '_cumulative_var'):
            self._cumulative_var[thresh] = i
        else:
            self._cumulative_var = {thresh : i}

    @requires_dataset
    def scree(self, save=True):

        var = self.pca.explained_variance_
        fig, ax = plt.subplots()
        sns.lineplot(x=range(1, len(var)+1),
                     y=var,
                     ax=ax)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Explained variance')
        ax.set_title('Scree plot')
        plt.show()
        if save == True:
            fig.savefig('scree.png', format='png')

    
    @requires_dataset
    def _calculate_metrics(self,
                           cumvar,
                           section:str=None):
        self._kmo(section)
        self._bartlett(section)

        self._kaiser(section)

        self.run(n_components=None, section=section)
        self.cumulative_var(cumvar)
        self.scree()


    def evaluate_models(self,
                        kaiser=True,
                        cumulative_variance:float=0.8,
                        section:str=None):
        
        self._calculate_metrics(cumvar=cumulative_variance, section=section)
        
        # Interpret Bartlett
        print('\n BARTLETT SPHERICITY TEST \n')
        print(f"Chi-squared: {self.bartlett['chi-squared']}")
        print(f"P-value: {self.bartlett['p-val']}")
        
        if self.bartlett['p-val'] > 0.05:
             print('P-value above 0.05, we advise not employ a PCA')
        else:
             print('P-value below 0.05, you may employ a PCA')

        # Interpret KMO
        print('\n KAISER-MEYER-OLKIN (KMO) \n')
        print(f"KMO: {self.kmo}")

        if self.kmo > 0.8:
            print('KMO above 0.8, the sampling is adequate.')
            print('You may employ PCA.')
        elif self.kmo > 0.5:
            print('KMO between 0.5 and 0.8, the sampling is not ideal.')
            print('But you may proceed with PCA.')
        else:
            print('KMO below 0.05, we advise not employ a PCA')
        
        # Interpret Kaiser
        print('\n KAISER CRITERION \n')
        print(f"Number of eigenvalues >1: {self.kaiser}")
 
        # Interpreting cumulative variance
        print('\n CUMULATIVE VARIANCE \n')
        n = self._cumulative_var[cumulative_variance]
        print(f'{n} components explain {100*cumulative_variance}% of the variance.')

        # Retrain
        if kaiser == True:
            n = self.kaiser
            print('\n Using the Kaiser criterion to determine the number of components\n')
        else:
            print(f'\n Considering a cumulative variance of {100*cumulative_variance}% to determine the number of components\n')
        print(f'Retraining model with {n} components...')
        self.run(n_components=n, section=section)


    @requires_dataset
    def plot(self,
             vectors=True,
             cmap:str='muted',
             legend=False,
             save:bool=True):

        legend = 'brief' if legend == True else False

        # PCA plot
        PC1_var= round(self.pca.explained_variance_ratio_[0] * 100, 2)
        PC2_var= round(self.pca.explained_variance_ratio_[1] * 100, 2)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7), dpi = 600)
        sns.scatterplot(data=self.coordinates,
                        x='PC1',
                        y='PC2',
                        hue='class',
                        palette=cmap,
                        ax=axes,
                        legend=legend)
        plt.xlabel(f'PC1 (explained variance: {PC1_var}%)')
        plt.ylabel(f'PC2 (explained variance: {PC2_var}%)')
        
        # Plot loadings vectors (arrows)
        if vectors == True:
            
            features = self.data_processor.dataset.get_X().columns
            loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
            
            for i, feature in enumerate(features):
                plt.arrow(0, 
                          0, 
                          loadings[i, 0]*3, 
                          loadings[i, 1]*3, 
                          alpha=0.5,
                          head_width=0.05
                          , length_includes_head=True)
                plt.text(loadings[i, 0]*3.2,
                         loadings[i, 1]*3.2,
                         feature,
                         size=10,
                         ha='center',
                         va='center')

        # Save the figure
        if save == True:
            fig.savefig('PCA.png', format='png')
        plt.show()
        plt.close(fig)
