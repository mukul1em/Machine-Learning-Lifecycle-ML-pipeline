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
    data = handImbalance.Smote(data)
    feat = FeatureSelection(data)
    data = feat.removeCorr()
    data = data.to_csv('./data versions/corr_feat.csv')
    data = pd.read_csv(r'./data versions/corr_feat.csv')
    cols = data.columns
    data = removeOutlier(data, cols)
    

    xtrain = data.iloc[:,:-1].values
    ytrain  = data.iloc[:, -1].values

    optimization_function = partial(
        tune_logistic,
        x=xtrain,
        y=ytrain
    
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_function, n_trials=15)

    print("best params")
    print("****************************")
    print(f"The best parameters are : \n{study.best_params['C']}")
    print(f"The best score are : \n{study.best_value}")

    print("****************************")


    mlflow.set_experiment("Best Logistics Model")
    # lr = Model.logistic()
    # scores = Training.cross_val(10, 3, model = lr)
    with mlflow.start_run():
        
        
        # mlflow.sklearn.log_model(lr, "logistic regression")
        mlflow.log_param("C", study.best_params["C"])
        mlflow.log_param("f1 scores", study.best_value)
        # mlflow.sklearn.autolog()
        # mlflow.set_tag("best run", )
    