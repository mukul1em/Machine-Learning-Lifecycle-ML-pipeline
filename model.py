from cv2 import solve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class Model:

    def logistic(C):
        lr = LogisticRegression(C=C)
        return lr
    
    def randomforest(n_estimators,max_depth, max_features , criterion):
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,max_features=max_features,criterion=criterion)
        return rf

    