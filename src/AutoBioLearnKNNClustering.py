import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from AutoBioLearnUnsupervisedLearning import AutoBioLearnUnsupervisedLearning
from decorators import requires_dataset

class AutoBioLearnKNNClustering(AutoBioLearnUnsupervisedLearning):
    
    def __init__(self) -> None:
        super().__init__()
        
    @requires_dataset
