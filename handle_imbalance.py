import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import RobustScaler

data = pd.read_csv(r'./data versions/training_data.csv')


class handImbalance:
    def __init__(self, data):
        self.data = data

    
    def undersample(self):
        ros = RandomUnderSampler(random_state=10)
        x = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values
        xresampled, y_resampled = ros.fit_resample(x, y)
        datay = pd.DataFrame(y_resampled)
        datax = pd.DataFrame(xresampled)
        data_under= pd.concat([datax, datay], axis=1)
        data_under.columns = data.columns
        rs_time = RobustScaler()
        rs_amount = RobustScaler()
        data_under['Amount'] = rs_amount.fit_transform(data_under['Amount'].values.reshape(-1, 1))
        data_under['Time'] = rs_time.fit_transform(data_under['Time'].values.reshape(-1,1))
        data_under.to_csv('./data versions/data_under.csv', index=False)
        return data_under
    
    def Smote(self):
        oversample = SMOTE()
        rs_time = RobustScaler()
        rs_amount = RobustScaler()
        data['Amount'] = rs_amount.fit_transform(data['Amount'].values.reshape(-1, 1))
        data['Time'] = rs_time.fit_transform(data['Time'].values.reshape(-1,1))
        x, y = data.iloc[:,:-1].values, data.iloc[:, -1].values
        x, y = oversample.fit_resample(x,y)
        datay = pd.DataFrame(y)
        datax = pd.DataFrame(x)
        data_over =  pd.concat([datax, datay], axis=1)
        data_over.columns = data.columns
        data_over.to_csv('./data versions/data_over.csv', index=False, header=data.columns)
        return data_over
    
    def randomoversampler(self):
        ros = RandomOverSampler(random_state=10)
        x = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values
        xresampled, y_resampled = ros.fit_resample(x, y)
        datay = pd.DataFrame(y_resampled)
        datax = pd.DataFrame(xresampled)
        data_under= pd.concat([datax, datay], axis=1)
        data_under.columns = data.columns
        rs_time = RobustScaler()
        rs_amount = RobustScaler()
        data_under['Amount'] = rs_amount.fit_transform(data_under['Amount'].values.reshape(-1, 1))
        data_under['Time'] = rs_time.fit_transform(data_under['Time'].values.reshape(-1,1))
        data_under.to_csv('./data versions/data_over.csv', index=False)
        return data_under
