from timeit import default_timer as timer

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import daal4py as d4p
import numpy as np
import pandas as pd

import common

NUM_LOOPS = 100
PARAMS = { 
    'objective': 'reg:squarederror',
    'alpha': 0.9,
    'max_bin': 256,
    'scale_pos_weight': 2,
    'learning_rate': 0.1, 
    'subsample': 1, 
    'reg_lambda': 1, 
    'min_child_weight': 0,
    'max_depth': 8, 
    'max_leaves': 2**8, 
    'tree_method': 'hist', 
    'predictor': 'cpu_predictor'
}

#gbt = d4p.gbt_regression_training(maxIterations=200)
TRAIN_DF = xgb.DMatrix(data=common.X, label=common.y)
MODEL = xgb.train(params=PARAMS, dtrain=TRAIN_DF)
#MODEL = gbt.compute(
#            pd.DataFrame(common.X, dtype=np.float32), 
#            pd.DataFrame(common.y, dtype=np.float32)).model

daal_model = d4p.get_gbt_model_from_xgboost(MODEL)

def run_inference(num_observations:int = 1000):
    """Run xgboost for specified number of observations"""
    # Load data
    test_df = common.get_test_data(num_observations)
    #test_df = common.get_test_data(num_observations)
    #data = pd.DataFrame(test_df, dtype=np.float32)
    #predictor = d4p.gbt_regression_prediction(**PARAMS)
    num_rows = len(test_df)

    run_times = []
    inference_times = []
    for _ in range(NUM_LOOPS):
        
        start_time = timer()
        daal_predict_algo = d4p.gbt_regression_prediction(fptype='float')
        daal_prediction = daal_predict_algo.compute(test_df, daal_model)
        #predictor.compute(data, MODEL)
        end_time = timer()

        total_time = end_time - start_time
        run_times.append(total_time*10e3)

        inference_time = total_time*(10e6)/num_rows
        inference_times.append(inference_time)

    return_elem = common.calculate_stats(inference_times)
    print(num_observations, ", ", return_elem)
    return return_elem