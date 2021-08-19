from timeit import default_timer as timer

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import daal4py as d4p
import numpy as np
import pandas as pd

import common
d4p.daalinit()
NUM_LOOPS = 50

print("Computing for Kmeans with Daal")


def run_inference(batch_size):
    """Run xgboost for specified number of observations"""
    # Load data
    test_df = create_batches(common.X_dfc,batch_size)
    ######################
    print("_______________________________________")
    run_times = []
    inference_times = []
    for _ in range(NUM_LOOPS):
        for b in test_df:

            start_time = timer()

            init_alg = d4p.kmeans_init(nClusters = 5, fptype = "float", method = "randomDense")
            centroids = init_alg.compute(b).centroids
            alg = d4p.kmeans(nClusters = 5, maxIterations = 100, fptype = "float", accuracyThreshold = 0,
                             assignFlag = False)
            result = alg.compute(b, centroids)

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