from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import common

kmeans_kwargs = {
   "init": "random",
   "n_init": 10,
   "max_iter": 100,
   "random_state": 42,
}

NUM_LOOPS = 100

print("Computing for KMeans Clustering Training without Daal")

cluster = KMeans(n_clusters=5, **kmeans_kwargs)
cluster.fit(common.X_df)

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

        cluster = KMeans(n_clusters=5, **kmeans_kwargs)
        cluster.fit(test_df)
        
        end_time = timer()

        total_time = end_time - start_time
        # run_times.append(total_time*10e3)

        inference_time = total_time*(10e6)/num_rows
        inference_times.append(inference_time)

    return_elem = common.calculate_stats(inference_times)
    print(num_observations, ", ", return_elem)
    return return_elem