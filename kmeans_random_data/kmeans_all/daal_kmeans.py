from timeit import default_timer as timer

import daal4py as d4p
import numpy as np
import pandas as pd

import common

d4p.daalinit()
NUM_LOOPS = 10

print("Computing for Kmeans with Daal")

init_alg = d4p.kmeans_init(nClusters = 10, fptype = "float", method = "randomDense")
centroids = init_alg.compute(common.X_df).centroids

def run_inference(num_observations:int = 1000):
    """Run xgboost for specified number of observations"""
    # Load data
    test_df = common.get_test_data_df(X=common.X_df,size = num_observations)
    num_rows = len(test_df)
    ######################
    print("_______________________________________")
    print("Total Number of Rows", num_rows)
    # run_times = []
    inference_times = []
    for _ in range(NUM_LOOPS):
        
        start_time = timer()
        
        alg = d4p.kmeans(nClusters = 10, maxIterations = 0, fptype = "float", accuracyThreshold = 0,
                         assignFlag = False)
        alg.compute(test_df, centroids)
        
        end_time = timer()

        total_time = end_time - start_time
        # run_times.append(total_time*10e3)

        inference_time = total_time*(10e6)/num_rows
        inference_times.append(inference_time)

    return_elem = common.calculate_stats(inference_times)
    print(num_observations, ", ", return_elem)
    return return_elem