from sklearn.datasets import load_boston
from sklearn.datasets import make_classification

import numpy as np
#import xgboost as xgb
import pandas as pd

# Load Boston Housing dataset
X, y = load_boston(return_X_y=True)

def boston_train():
    boston_pd_x = pd.DataFrame(load_boston().data)
    boston_pd_y = pd.DataFrame(y)
    return boston_pd_x,boston_pd_y

X_df = pd.DataFrame(load_boston().data)
y_df = pd.DataFrame(y)
# Label for the output

X_cls, y_cls = make_classification(n_samples=1000, n_features=20, n_repeated=0, n_classes=2)

X_dfc = pd.DataFrame(X_cls)
y_dfc = pd.DataFrame(y_cls)

STATS = '#, median, mean, std_dev, min_time, max_time, quantile_10, quantile_90'


def get_test_data(size: int = 1):
    """Generates a test dataset of the specified size""" 
    num_rows = len(X)
    test_df = X.copy()

    while num_rows < size:
        test_df = np.append(test_df, test_df, axis=0)
        num_rows = len(test_df)

    return test_df[:size]
    
def get_test_data_y(size: int = 1):
    """Generates a test dataset of the specified size""" 
    num_rows = len(y)
    test_df = y.copy()

    while num_rows < size:
        test_df = np.append(test_df, test_df, axis=0)
        num_rows = len(test_df)

    return test_df[:size]

def get_test_data_yc(size: int = 1):
    """Generates a test dataset of the specified size""" 
    num_rows = len(y_cls)
    test_df = y_cls.copy()

    while num_rows < size:
        test_df = np.append(test_df, test_df, axis=0)
        num_rows = len(test_df)

    return test_df[:size]

def get_test_data_df(X,size: int = 1):
    """Generates a test dataset of the specified size""" 
    num_rows = len(X)
    test_df = X.copy()

    while num_rows < size:
        test_df = test_df.append(test_df)
        num_rows = len(test_df)

    return test_df[:size].reset_index(drop = True)

def calculate_stats(time_list):
    """Calculate mean and standard deviation of a list"""
    time_array = np.array(time_list)

    median = np.median(time_array)
    mean = np.mean(time_array)
    std_dev = np.std(time_array)
    max_time = np.amax(time_array)
    min_time = np.amin(time_array)
    quantile_10 = np.quantile(time_array, 0.1)
    quantile_90 = np.quantile(time_array, 0.9)
    
    basic_key = ["median","mean","std_dev","min_time","max_time","quantile_10","quantile_90"]
    basic_value = [median,mean,std_dev,min_time,max_time,quantile_10,quantile_90]

    dict_basic = dict(zip(basic_key, basic_value))

    
    return pd.DataFrame(dict_basic, index = [0])