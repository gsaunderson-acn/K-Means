from timeit import default_timer as timer

import xgboost as xgb

import common
import gc

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
TRAIN_DF = xgb.DMatrix(data=common.X, label=common.y)
MODEL = xgb.train(params=PARAMS, dtrain=TRAIN_DF)


def run_inference(num_observations:int = 1000):
    """Run xgboost for specified number of observations"""
    # Load data
    test_df = common.get_test_data(num_observations)

    num_rows = len(test_df)
    # print(f"Running {NUM_LOOPS} inference loops with batch size {num_rows}...")

    run_times3 = []
    inference_times3 = []
    for _ in range(NUM_LOOPS):
        start_time = timer()
        data = xgb.DMatrix(test_df)
        MODEL.predict(data)
        end_time = timer()
		
        total_time3 = end_time - start_time
        run_times3.append(total_time3*10e3)

        inference_time3 = total_time3*(10e6)/num_rows
        inference_times3.append(inference_time3)
		
    print(num_observations, ", ", common.calculate_stats(inference_times3))