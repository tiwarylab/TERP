"""
TERP: Thermodynamically Explainable Representations of AI and other black-box Paradigms
"""
import numpy as np
import os
import sys
import sklearn.metrics as met
import logging
import time
import pandas as pd
from tqdm import tqdm
import copy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.linear_model import Ridge
import pickle

results_directory = 'TERP_results'
os.makedirs(results_directory, exist_ok = True)
rows = 'null'
neighborhood_data = 'null'
############################################
# Set up logger
fmt = '%(asctime)s %(name)-15s %(levelname)-8s %(message)s'
datefmt='%m-%d-%y %H:%M:%S'
logging.basicConfig(level=logging.INFO,format=fmt,datefmt=datefmt,filename=results_directory+'/TERP_2.log',filemode='w')
logger1 = logging.getLogger('initialization')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(fmt,datefmt=datefmt)
console_handler.setFormatter(formatter)
logger1.addHandler(console_handler)
print(100*'-')
logger1.info('Starting TERP...')
print(100*'-')
logger2 = logging.getLogger('TERP_SGD_step_02')
console_handler.setFormatter(formatter)
logger2.addHandler(console_handler)
############################################

if '-TERP_input' in sys.argv:
  TERP_input = np.load(sys.argv[sys.argv.index('-TERP_input') + 1])
  rows = TERP_input.shape[0]
  k_max = TERP_input.shape[1]
  neighborhood_data = TERP_input.reshape(rows,-1)
  logger1.info('Input data read successful ...')

if '-blackbox_prediction' in sys.argv:
  pred_proba = np.load(sys.argv[sys.argv.index('-blackbox_prediction') + 1])
  if pred_proba.shape[0] != rows:
    logger1.error('TERP input and blackbox prediction probability dimension mismatch!')
    raise Exception()
  pred_proba = pred_proba.reshape(rows,-1)
else:
  logger1.error('Missing blackbox prediction!')
  raise Exception()

if '-explain_class' in sys.argv:#explain class is useful for independent classes for example multi-class images
  explain_class = int(sys.argv[sys.argv.index('-explain_class') + 1])
  logger1.info("Toatal number of classes :: " + str(pred_proba.shape[1]))
  logger1.info('explain_class :: ' + str(explain_class))
  if explain_class not in [i for i in range(pred_proba.shape[1])]:
    logger1.error('Invalid -explain_class!')
    raise Exception()
else:
  explain_class = np.argmax(pred_proba[0,:])
  logger1.warning('explain_class not provided, defaulting to class with maximum predictiion probability :: ' + str(explain_class))

target = pred_proba[:,explain_class]

def similarity_kernel(data, kernel_width):
  distances = met.pairwise_distances(data,data[0].reshape(1, -1),metric='euclidean').ravel()
  return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

if '-cutoff' in sys.argv:
  cutoff = float(sys.argv[sys.argv.index('-cutoff') + 1])
  logger1.info("Provided cutoff :: " + str(cutoff))
else:
  cutoff = 0.8
  logger1.warning('Cutoff not provided. Defaulting to :: ' + str(cutoff))

if '--euclidean' in sys.argv:
  weights = similarity_kernel(neighborhood_data, 0.75*np.sqrt(neighborhood_data.shape[1]))
  logger1.info("Euclidean distance flag provided. Computing euclidean distance over the entire input distance for similarity measure!")
  
elif '--cosine_d' in sys.argv:
  weights = np.sqrt(np.exp(-(met.pairwise.cosine_distances(neighborhood_data,neighborhood_data[0,:].reshape(1, -1)).ravel()** 2)/0.25**2))
  logger1.info("cosine_d distance flag provided. Computing cosine distance for similarity measure (appropriate for image or text data)!")
 
else:
  threshold, upper, lower = 0.5, 1, 0
  target_binarized = np.where(target>threshold, upper, lower)

  clf = lda()
  clf.fit(neighborhood_data,target_binarized)
  projected_data = clf.transform(neighborhood_data)
  weights = similarity_kernel(projected_data, 1)
  logger1.info("No distance flag provided. Performing 1-d LDA projection to compute similarity measure!")

if '-selected_features' in sys.argv:
  feat_dir = sys.argv[sys.argv.index('-selected_features') + 1]
  with open(feat_dir, "rb") as fp:   # Unpickling
     feat_desc = pickle.load(fp)
  selected_features = np.array(feat_desc[0])
  tot_feat = feat_desc[1]
  logger1.info("Feature selection results read successful!")
else:
  logger1.error('Missing selected features!')
  raise Exception

predict_proba = pred_proba[:,explain_class]
data = neighborhood_data*(weights**0.5).reshape(-1,1)
labels = target.reshape(-1,1)*(weights.reshape(-1,1)**0.5)

corr_coef_master = []

def SGDreg(predict_proba, data, labels):
  clf = Ridge(alpha=1.0, random_state = 10, solver = 'sag')
  clf.fit(data,labels.ravel())
  coefficients = clf.coef_
  intercept = clf.intercept_
  return coefficients, intercept


def unfaithfulness_calc(k, N, data, predict_proba, best_parameters_master):
  models = []
  TERP_SGD_parameters = []
  TERP_SGD_unfaithfulness = []
  if k == 1:
    inherited_nonzero = np.array([],dtype=int)
    inherited_zero = np.arange(N)

  elif k > 1:
    inherited_nonzero = np.nonzero(best_parameters_master[k-2][:-1])[0]
    inherited_zero = np.where(best_parameters_master[k-2][:-1] == 0)[0]

  for i in range(N-k+1):
    models.append(np.append(inherited_nonzero, inherited_zero[i]))
    result_a, result_b = SGDreg(predict_proba, data[:,models[i]], labels)
    parameters = np.zeros((N+1))
    parameters[models[i]] = result_a
    parameters[-1] = result_b
    TERP_SGD_parameters.append(parameters)
    residual = np.absolute(labels - (np.column_stack((data, np.ones((data.shape[0]))))@parameters[:]).reshape(-1,1))**2
    TERP_SGD_unfaithfulness.append(np.sum(residual))
  best_model = np.argsort(TERP_SGD_unfaithfulness)[0]
  best_parameters_master.append(TERP_SGD_parameters[best_model])
  
  temp_coef_1 = TERP_SGD_parameters[best_model][:-1]
  temp_coef_2 = np.zeros((tot_feat))
  temp_coef_2[selected_features] = copy.deepcopy(temp_coef_1)
  best_parameters_converted.append(temp_coef_2)

  best_unfaithfulness_master.append(TERP_SGD_unfaithfulness[best_model])

  surrogate_pred = data@TERP_SGD_parameters[best_model][:-1]
  corr_coef_master.append(np.corrcoef(labels.flatten(),surrogate_pred)[0,1])

  logger1.info("Unfaithfulness :: " + str(TERP_SGD_unfaithfulness[best_model]))
  logger1.info("Relevant features :: " + str(selected_features[np.nonzero(TERP_SGD_parameters[best_model][:-1])[0]]))
best_parameters_master = []
best_parameters_converted = []
best_unfaithfulness_master = []

N = data.shape[1]
k_array = np.arange(1,k_max + 1)

logger1.info('Similarity computation complete...')
print(100*'-')
print(100*'-')
starttime = time.time()
for k in k_array:
  print(100*'-')
  logger2.info('Scanning models for k :: ' + str(k))
  print(100*'-')
  unfaithfulness_calc(k, N, data, predict_proba, best_parameters_master)


np.save(results_directory + '/weights_final.npy', weights)
np.save(results_directory + '/coefficients_final.npy', np.array(best_parameters_converted))
np.save(results_directory + '/unfaithfulness_final.npy', np.array(best_unfaithfulness_master))
np.save(results_directory + '/corr_coef_final.npy', np.array(corr_coef_master))

## S, and zeta calculations
delta = 0.1
S = np.log(k_array)
def term(alpha,S, U):
  return alpha*np.array(S) + U
min_sol = []
min_sol_val = []
stop = False
i=copy.deepcopy(delta)
scan = []
while stop==False:
  zeta = term(i,S,best_unfaithfulness_master)
  zeta_loc = np.argmin(zeta)
  min_sol.append(zeta_loc)
  min_sol_val.append(zeta[zeta_loc])
  scan.append(i)
  i+=delta
  if zeta_loc==0:
    break

np.save(results_directory + '/temp_zeta_kc.npy', np.column_stack((scan, min_sol_val, np.array(min_sol))))
counts = np.bincount(min_sol)
print(100*'-')
if np.argmax(counts)+1==k_array[-1]:
  logger1.info('WARNING! Most relevant features obtained at highest k = ' + str(k_array[-1]) + ' model. Consider increasing cut-off in last step!')
else:
  logger1.info('Most relevant features obtained at k = ' + str(np.argmax(counts)+1) + ' model!')

####
endtime = time.time()
monte_carlo_time = endtime - starttime
print(100*'-')
logger2.info('computation time :: ' + str(int(monte_carlo_time/60)) + ' min ' + "{:.3f}".format(monte_carlo_time%60) + ' sec...')
print(100*'-')
