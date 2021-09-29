from sklearn.datasets import make_blobs

import numpy as np
import pandas as pd

X, y = make_blobs(n_samples = 10 * 10**6, n_features = 20,
                  centers = 10, cluster_std = 0.2,
                  center_box = (-10.0, 10.0), random_state = 777)

X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)

STATS = '#, median, mean, std_dev, min_time, max_time, quantile_10, quantile_90'

def get_test_data_df(X,size: int = 1):
    """Generates a test dataset of the specified size""" 
    
    return X[:size].reset_index(drop = True)

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