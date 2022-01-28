from tkinter.filedialog import LoadFileDialog
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from remove_outlier import removeOutlier
from feature_selection import FeatureSelection
from model import Model
import statsmodels.regression.linear_model as sm
from mlflow import log_metric
import mlflow
from functools import partial
import numpy as np
from sklearn import model_selection
import optuna
from sklearn import metrics
from hyperparamter_tune import tune_logistic, tune_rf
from handle_imbalance import handImbalance
from datadrift import datadrift

class Training:
    def __init__(self, xtrain, xtest, ytrain, ytest):
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
    
    def cross_val(nsplits, nrepeats, model):
        cv = RepeatedStratifiedKFold(n_splits=nsplits, n_repeats=nrepeats, random_state=10)
        scores = cross_val_score(model, xtrain, ytrain, cv=cv, n_jobs=-1, scoring='f1')
        return scores.mean()
    





if __name__ == '__main__':
    

    data = pd.read_csv(r'./data versions/data_over.csv')
    datadrift(data, data)
    data = handImbalance.Smote(data)
    feat = FeatureSelection(data)
    data = feat.removeCorr()
    data = data.to_csv('./data versions/corr_feat.csv')
    data = pd.read_csv(r'./data versions/corr_feat.csv')
    cols = data.columns
    data = removeOutlier(data, cols)
    

    xtrain = data.iloc[:,:-1].values
    ytrain  = data.iloc[:, -1].values

    
   
    lr = Model.logistic(C=0.8)
    scores = Training.cross_val(10, 3, model = lr)
    
        
        
    