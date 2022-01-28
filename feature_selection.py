import pandas as pd
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
import statsmodels.regression.linear_model as sm

class FeatureSelection:
    def __init__(self, data):
        self.data = data

    def removeCorr(self):
        """
        """
        corr = self.data.corr()
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= 0.9:
                    print(True)
                    if columns[j]:
                        columns[j] = False
        selected_columns = self.data.columns[columns]
        data = self.data[selected_columns]
        return data

    
    def anova(self,k):
        """
        """
        X, y = self.data.iloc[:,:-1], self.data.iloc[:, -1]
        fs = SelectKBest(score_func = f_classif, k=k)
        fs.fit(X,y)
        x = fs.transform(X)
        datax = pd.DataFrame(x)
        datax.columns = ['V_' + str(i+1) for i in range(x.shape[1])]
        datay = pd.DataFrame(y)
        data = pd.concat([datax, datay], axis=1)
        return data

    def mutual_info(self,k):
        """
        """
        X, y = self.data.iloc[:,:-1], self.data.iloc[:, -1]
        fs = SelectKBest(score_func = mutual_info_classif, k=k)
        fs.fit(X,y)
        x = fs.transform(X)
        datax = pd.DataFrame(x)
        datax.columns = ['V_' + str(i+1) for i in range(x.shape[1])]
        datay = pd.DataFrame(y)
        data = pd.concat([datax, datay], axis=1)
        return data

    def backwardElimination(self, sl):
        """
        """
        columns = self.data.columns[:-1].values
        x = self.data.iloc[:,:-1].values
        Y = self.data.iloc[:,-1].values
        numVars = len(x[0])
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(Y, x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            if maxVar > sl:
                for j in range(0, numVars - i):
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                        x = np.delete(x, j, 1)
                        columns = np.delete(columns, j)
        
        datax = pd.DataFrame(x,columns=columns)
        datay = self.data.iloc[:,-1]
        data = pd.concat([datax, datay], axis=1)
        return data

