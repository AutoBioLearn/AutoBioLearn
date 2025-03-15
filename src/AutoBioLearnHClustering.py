# -*- coding: utf-8 -*-

from AutoBioLearnUnsupervisedLearning import AutoBioLearnUnsupervisedLearning

class AutoBioLearnHClustering(AutoBioLearnUnsupervisedLearning):
    
    def execute_models(self, models:list[str]=[], params={}, section:str=None):
            
        self.data_processor.dataset
