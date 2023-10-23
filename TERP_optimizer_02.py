"""
TERP: Thermodynamically Explainable Representations of AI and other black-box Paradigms
"""
import numpy as np
import os
import sys
import sklearn.metrics as met
import logging
import time
from tqdm import tqdm
import copy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.linear_model import Ridge
import pickle

results_directory = 'TERP_results_2'
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
if '--nolog' in sys.argv:
  logger1.propagate = False
  logger2.propagate = False
############################################

if '-TERP_input' in sys.argv:
  TERP_input = np.load(sys.argv[sys.argv.index('-TERP_input') + 1])
  rows = TERP_input.shape[0]
  neighborhood_data = TERP_input.reshape(rows,-1)
  logger1.info('Input data read successful ...')

if '-unf_dec_threshol' in sys.argv:
  unf_threshold = float(sys.argv[sys.argv.index('-TERP_input') + 1])
else:
  unf_threshold = 0.01

if '-blackbox_prediction' in sys.argv:
  pred_proba = np.load(sys.argv[sys.argv.index('-blackbox_prediction') + 1])
  if pred_proba.shape[0] != rows:
    logger1.error('TERP input and blackbox prediction probability dimension mismatch!')
    raise Exception()
  pred_proba = pred_proba.reshape(rows,-1)
else:
  logger1.error('Missing blackbox prediction!')
  raise Exception()

if '--save_all' in sys.argv:
  save_all = True
  logger1.info('All files will be saved!')
else:
  save_all = False

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
  with open(feat_dir, "rb") as fp:
     feat_desc = pickle.load(fp)
  selected_features = np.array(feat_desc[0])
  neighborhood_data = neighborhood_data[:, selected_features]
  k_max = neighborhood_data.shape[1]
  tot_feat = feat_desc[1]
  logger1.info("Feature selection results read successful!")
else:
  logger1.error('Missing selected features!')
  raise Exception

predict_proba = pred_proba[:,explain_class]
data = neighborhood_data*(weights**0.5).reshape(-1,1)
labels = target.reshape(-1,1)*(weights.reshape(-1,1)**0.5)

def SGDreg(predict_proba, data, labels):
  clf = Ridge(alpha=1.0, random_state = 10, solver = 'sag')
  clf.fit(data,labels.ravel())
  coefficients = clf.coef_
  intercept = clf.intercept_
  return coefficients, intercept

def interp(coef_array):
  a = np.absolute(coef_array)/np.sum(np.absolute(coef_array))
  t = 0
  for i in range(a.shape[0]):
    if a[i]==0:
      continue
    else:
      t += a[i]*np.log(a[i])
  return -t/np.log(coef_array.shape[0])

def unfaithfulness_calc(k, N, data, predict_proba, best_parameters_master):
  models = []
  TERP_SGD_parameters = []
  TERP_SGD_unfaithfulness = []
  TERP_SGD_interp = []
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
    residual = np.corrcoef(labels[:,0],(np.column_stack((data, np.ones((data.shape[0]))))@parameters[:]).reshape(-1,1)[:,0])[0,1]
    TERP_SGD_unfaithfulness.append(1-np.absolute(residual))
    TERP_SGD_interp.append(interp(TERP_SGD_parameters[-1][:-1]))
    TERP_SGD_IFE = np.array(TERP_SGD_unfaithfulness)

  if save_all == True:
    np.save(results_directory + '/' + str(k) + '_feature_coefficients.npy', TERP_SGD_parameters)
    np.save(results_directory + '/' + str(k) + '_interpretation_entropy.npy', TERP_SGD_interp)
    np.save(results_directory + '/' + str(k) + '_unfaithfulness_scores.npy', TERP_SGD_unfaithfulness)

  best_model = np.argsort(TERP_SGD_IFE)[0]
  best_parameters_master.append(TERP_SGD_parameters[best_model])
  best_interp_master.append(TERP_SGD_interp[best_model])

  temp_coef_1 = TERP_SGD_parameters[best_model][:-1]
  temp_coef_2 = np.zeros((tot_feat))
  temp_coef_2[selected_features] = copy.deepcopy(temp_coef_1)
  best_parameters_converted.append(temp_coef_2)
  best_unfaithfulness_master.append(TERP_SGD_unfaithfulness[best_model])

  surrogate_pred = data@TERP_SGD_parameters[best_model][:-1]

best_parameters_master = []
best_parameters_converted = []
best_unfaithfulness_master = []
best_interp_master = []

N = data.shape[1]
k_array = np.arange(1,k_max + 1)

logger1.info('Similarity computation complete...')
print(100*'-')

starttime = time.time()
for k in tqdm(k_array, desc="Number of models constructed:: "):
  unfaithfulness_calc(k, N, data, predict_proba, best_parameters_master)

np.save(results_directory + '/neighborhood_similarity_final.npy', weights)
np.save(results_directory + '/feature_coefficients_final.npy', np.array(best_parameters_converted))
np.save(results_directory + '/unfaithfulness_scores_final.npy', np.array(best_unfaithfulness_master))
np.save(results_directory + '/interpretation_entropy_final.npy', np.array(best_interp_master))

def zeta(U,S,theta):
  return U + theta*S

optimal_k = 1
import copy

def charac_theta(d_U,d_S):
  return -d_U/d_S

if N<=3:
  for i in range(1,N):
    if best_unfaithfulness_master[i]<=best_unfaithfulness_master[i-1] - unf_threshold:
      prime_model = copy.deepcopy(i)-2
      continue
    else:
      print('k ::', prime_model+3, 'is the best model')
      break

else:
  charac_theta_mast = []

  d_U_lst = []
  d_S_lst = []
  for i in range(1,selected_features.shape[0]):
    d_U_lst.append(best_unfaithfulness_master[i] - best_unfaithfulness_master[i-1])
    d_S_lst.append(best_interp_master[i] - best_interp_master[i-1])

  for i in range(selected_features.shape[0]-1):
    charac_theta_mast.append(charac_theta(d_U_lst[i], d_S_lst[i]))

  range_theta_mast = []
  for i in range(1,len(charac_theta_mast)):
    range_theta_mast.append(np.array(charac_theta_mast)[i]-np.array(charac_theta_mast)[i-1])

  prime_model = np.argmin(np.array(range_theta_mast))

np.save(results_directory + '/optimal_feature_weights.npy', np.absolute(np.array(best_parameters_converted)[prime_model+2])/np.sum(np.absolute(np.array(best_parameters_converted)[prime_model+2])))
optimal_scores = np.array([best_unfaithfulness_master[prime_model+2], best_interp_master[prime_model+2]])
np.save(results_directory + '/optimal_scores_unfaithfulness_interpretation_entropy.npy', optimal_scores)
if N>3:
  np.save(results_directory + '/charac_theta.npy', charac_theta_mast)
  np.save(results_directory + '/range_theta.npy', range_theta_mast)
####
endtime = time.time()
monte_carlo_time = endtime - starttime
logger2.info('Analysis complete! Computation time :: ' + str(int(monte_carlo_time/60)) + ' min ' + "{:.3f}".format(monte_carlo_time%60) + ' sec...')
print(100*'-')
