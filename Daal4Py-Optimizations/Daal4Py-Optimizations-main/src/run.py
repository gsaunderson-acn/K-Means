import argparse
import logging
import pandas as pd
from pathlib import Path

from timeit import default_timer as timer
import common


# Setup logging
# logger = logging.getLogger('__name__')
# logging.getLogger().setLevel(logging.INFO)


# Initialize Command Line arguments
parser = argparse.ArgumentParser(description='Runs inference on models.')
parser.add_argument(
    '-m', '--model', type=str, default="xgboost",
    help="Type of model to run")
parser.add_argument(
    '-o', '--observations', type=int, default=1e6,
    help="Max batch size")
args = parser.parse_args()


# main module
if __name__=='__main__':

    # Load data
    model = args.model
    logging.info(f"Running benchmark for {model}...")

    # Run Inferencing
    if model == 'xgboost':
        from xg_all import xg
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            xg.run_inference(batch_size)
            batch_size *= 10
            
    elif model == 'daal':
        from xg_all import daal
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = daal.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)  
    elif model == 'daal_lm':
        from lm_all import daal_lm
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = daal_lm.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)

    elif model == 'daal_lm_training':
        from lm_all import daal_lm_training
        temp_df = pd.DataFrame()
        batch_size = 1000  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = daal_lm_training.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
            
    elif model == 'daal_rf':
        from rf_all import daal_rf
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = daal_rf.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
  
    elif model == 'daal_rf_training':
        from rf_all import daal_rf_training
        temp_df = pd.DataFrame()
        batch_size = 1000  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = daal_rf_training.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
          
    elif model == 'lm':
        from lm_all import lm
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = lm.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)

    elif model == 'lm_training':
        from lm_all import lm_training
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = lm_training.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
            
    elif model == 'rf':
        from rf_all import rf
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = rf.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)

    elif model == 'rf_training':
        from rf_all import rf_training
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = rf_training.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
            
    elif model == 'daal_logit':
        from logit_all import daal_logit
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = daal_logit.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)

    elif model == 'daal_logit_training':
        from logit_all import daal_logit_training
        temp_df = pd.DataFrame()
        batch_size = 1000  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = daal_logit_training.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
            
    elif model == 'logit':
        from logit_all import logit
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = logit.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)

    elif model == 'logit_patch':
        from logit_all import logit_patch
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = logit_patch.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)

    elif model == 'logit_patch_training':
        from logit_all import logit_patch_training
        temp_df = pd.DataFrame()
        batch_size = 100  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = logit_patch_training.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)


    elif model == 'logit_training':
        from logit_all import logit_training
        temp_df = pd.DataFrame()
        batch_size = 100  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = logit_training.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
        
    elif model == 'lm_patch':
        from lm_all import lm_patch
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = lm_patch.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
        
    elif model == 'rf_patch':
        from rf_all import rf_patch
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = rf_patch.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)

    elif model == 'rf_patch_training':
        from rf_all import rf_patch_training
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = rf_patch_training.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
        
    elif model == 'lm_patch_training':
        from lm_all import lm_patch_training
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = lm_patch_training.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
        
    elif model == 'kmeans':
        from kmeans_all import kmeans
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = kmeans.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
        
    elif model == 'kmeans_patch':
        from kmeans_all import kmeans_patch
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = kmeans_patch.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
        
    elif model == 'daal_kmeans_training':
        from kmeans_all import daal_kmeans_training
        temp_df = pd.DataFrame()
        batch_size = 100  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = daal_kmeans_training.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
        
    elif model == 'kmeans_training':
        from kmeans_all import kmeans_training
        temp_df = pd.DataFrame()
        batch_size = 100  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = kmeans_training.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
        
    elif model == 'kmeans_patch_training':
        from kmeans_all import kmeans_patch_training
        temp_df = pd.DataFrame()
        batch_size = 100  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = kmeans_patch_training.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
        
    elif model == 'daal_kmeans':
        from kmeans_all import daal_kmeans
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = daal_kmeans.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
        
    elif model == 'dbs':
        from dbs_all import dbs
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = dbs.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)    
        
    elif model == 'dbs_patch':
        from dbs_all import dbs_patch
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = dbs_patch.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df) 
     
    else:
        print(f"Could not find benchmark for {model}")
        # print(f"Available choices are: xgboost, daal")


    

