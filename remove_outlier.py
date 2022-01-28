import numpy as np 



def removeOutlier(data, cols):
    print(cols)
    Q1 = data[cols].quantile(0.25)
    Q3 = data[cols].quantile(0.75)
    IQR = Q3 - Q1
    condition =  ~((data[cols] < (Q1 - 3* IQR)) | (data[cols] > (Q3 + 3 * IQR))).any(axis=1)
    filtered_data = data[condition]
    return filtered_data

    

