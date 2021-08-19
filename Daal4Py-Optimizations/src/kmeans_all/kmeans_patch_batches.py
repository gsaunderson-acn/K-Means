from timeit import default_timer as timer

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import daal4py as d4p
import numpy as np
import pandas as pd
import common



import daal4py.sklearn
daal4py.sklearn.patch_sklearn()
from sklearn.cluster import KMeans

kmeans_kwargs = {
   "init": "random",
   "n_init": 10,
   "max_iter": 100,
   "random_state": 42,
}

NUM_LOOPS = 5

print("Computing for KMeans Clustering with Daal patch")

cluster = KMeans(n_clusters=5, **kmeans_kwargs)
cluster.fit(common.X_dfc)

def run_inference(batch_size):
    """Run xgboost for specified number of observations"""
    ######################
    print("_______________________________________")
    test_df = create_batches(common.X_dfc, batch_size)
    run_times = []
    inference_times = []

    for b in test_df:
        start_time = timer()

        predict_result = cluster.predict(b)
        #predictor.compute(data, MODEL)
        end_time = timer()

        total_time = end_time - start_time
        run_times.append(total_time*10e3)

        inference_time = total_time*(10e6)/batch_size
        inference_times.append(inference_time)

    return_elem = common.calculate_stats(inference_times)
    print(batch_size, ", ", return_elem)
    return return_elem


def create_batches(X, batch_size):
    batches = []
    for i in range(0, len(X), batch_size):
        batches.append(X[i:i+batch_size])
    return batches

