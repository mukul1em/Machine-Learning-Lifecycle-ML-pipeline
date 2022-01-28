import pandas as pd
from sklearn import datasets
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

def datadrift(cuurent_data, reference_data):
    data_drift = Dashboard(tabs=[DataDriftTab()])
    data_drift.calculate(cuurent_data, reference_data)
    data_drift.save('./templates/my_report.html')