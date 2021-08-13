# Daal4Py Optimization
Dataset obtained from :  
from sklearn.datasets import load_boston  
from sklearn.datasets import make_classification  

# Steps to Run on Intel Instance:
conda create -n test -y python=3.7  
conda activate test  
conda install -c conda-forge -y daal4py  
conda install -c conda-forge -y scikit-learn  
pip install xgboost  
conda install -c conda-forge -y pandas  
git clone  https://github.com/basilsony/Daal4Py-Optimizations  
cd Daal4Py-Optimizations/src/  
python3 run.py -m="Model Name" -o=100000  

# Steps to Run on Graviton Instance:
sudo snap install cmake --classic  
sudo apt update && sudo apt install -y python3-pip python3-pandas python3-sklearn  
pip3 install xgboost  
git clone  https://github.com/basilsony/Daal4Py-Optimizations  
cd Daal4Py-Optimizations/src/  
python3 run.py -m="Model Name" -o=100000

# Values within Model Name (-m) parameter:
Linear Regression - lm, lm_training, lm_patch, lm_patch_training, daal_lm, daal_lm_training  
logistic Regression - logit, logit_training, logit_patch, logit_patch_training, daal_logit, daal_logit_training  
Random Forest - rf, rf_training, rf_patch, rf_patch_training, daal_rf, daal_rf_training  
K-Means - kmeans, kmeans_training, kmeans_patch, kmeans_patch_training, daal_kmeans_training  
DBSCAN - dbs, dbs_patch  

