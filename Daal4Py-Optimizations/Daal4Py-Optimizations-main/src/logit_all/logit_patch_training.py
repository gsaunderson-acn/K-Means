from timeit import default_timer as timer

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import daal4py as d4p
import numpy as np
import pandas as pd
import common


NUM_LOOPS = 100

import daal4py.sklearn
daal4py.sklearn.patch_sklearn()
from sklearn.linear_model import LogisticRegression


def run_inference(num_observations:int = 1000):
    """Run xgboost for specified number of observations"""
    # Load data
    train_x_df = common.get_test_data_df(X=common.X_dfc,size = num_observations)
    train_y = common.get_test_data_yc(size = num_observations)	
    num_rows = len(train_x_df)
    ######################
    print("_______________________________________")
    print("Total Number of Rows", num_rows)
    run_times = []
    inference_times = []
    for _ in range(NUM_LOOPS):
        
        start_time = timer()
        reg = LogisticRegression().fit(train_x_df, train_y)
        end_time = timer()

        total_time = end_time - start_time
        run_times.append(total_time*10e3)

        inference_time = total_time*(10e6)/num_rows
        inference_times.append(inference_time)

    return_elem = common.calculate_stats(inference_times)
    print(num_observations, ", ", return_elem)
    return return_elem