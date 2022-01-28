import optuna
from model import Model
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import cross_val_score

def tune_logistic(trial, x, y):
    # penalty = trial.suggest_categorical("penalty",['l1', 'l2', 'elasticnet'])
    C = trial.suggest_uniform("C", 0.01, 1.0)
    # l1_ratio = trial.suggest_uniform("l1_ratio", 0.01, 1.0)
    model = Model.logistic( C=C)

    cv = model_selection.RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=10)
    scores = cross_val_score(model, x, y, cv=cv, n_jobs=-1, scoring='f1')
    
    return -1.0*np.mean(scores)  #-1 * mean because we have to minimize (for accuracy score) but for log loss we don't have to 


def tune_rf(trial, x, y):
    criterion  = trial.suggest_categorical("criterion",['gini','entropy'])
    n_estimators = trial.suggest_int("n_estimators",100,1500) #min and max values
    max_depth = trial.suggest_int("max_depth", 3, 15) 
    max_features = trial.suggest_uniform("max_features", 0.01, 1.0)

    model = Model.randomforest(
        n_estimators=n_estimators, max_depth=max_depth,
        max_features=max_features,criterion=criterion
    )
    cv = model_selection.RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=10)
    scores = cross_val_score(model, x, y, cv=cv, n_jobs=-1, scoring='f1')
    
    return -1.0*np.mean(scores)

