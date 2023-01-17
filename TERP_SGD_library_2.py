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


results_directory = 'TERP_results'
os.makedirs(results_directory, exist_ok = True)
rows = 'null'
neighborhood_data = 'null'
############################################
# Set up logger
fmt = '%(asctime)s %(name)-15s %(levelname)-8s %(message)s'
datefmt='%m-%d-%y %H:%M:%S'
logging.basicConfig(level=logging.INFO,format=fmt,datefmt=datefmt,filename=results_directory+'/TERP.log',filemode='w')
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

if '-TERP_categorical' in sys.argv:
  TERP_categorical = np.load(sys.argv[sys.argv.index('-TERP_categorical') + 1])
  rows = TERP_categorical.shape[0]
  TERP_categorical = TERP_categorical.reshape(rows,-1)
  neighborhood_data = TERP_categorical
  logger1.info('Input includes categorical data')

if '-TERP_numeric' in sys.argv:
  TERP_numeric = np.load(sys.argv[sys.argv.index('-TERP_numeric') + 1])
  logger1.info('Input includes numerical data')
  if rows == 'null':
    rows = TERP_numeric.shape[0]
    TERP_numeric = TERP_numeric.reshape(rows,-1)
    neighborhood_data = TERP_numeric
  else:
    if rows != TERP_numeric.shape[0]:
      logger1.error('Input (numeric) dimension mismatch!')
      raise Exception()
    TERP_numeric = TERP_numeric.reshape(rows,-1)
    neighborhood_data = np.column_stack((neighborhood_data, TERP_numeric))

if '-TERP_periodic' in sys.argv:
  TERP_periodic = np.load(sys.argv[sys.argv.index('-TERP_periodic') + 1])
  logger1.info('Input includes periodic data')
  if rows == 'null':
    rows = TERP_periodic.shape[0]
    TERP_periodic = TERP_periodic.reshape(rows,-1)
    neighborhood_data = TERP_periodic
  else:
    if rows != TERP_periodic.shape[0]:
      logger1.error('Input (periodic) dimension mismatch!')
      raise Exception()
    TERP_periodic = TERP_periodic.reshape(rows,-1)
    neighborhood_data = np.column_stack((neighborhood_data, TERP_periodic))

if '-TERP_sin_cos' in sys.argv:
  TERP_sin_cos = np.load(sys.argv[sys.argv.index('-TERP_sin_cos') + 1])
  if rows == 'null':
    rows = TERP_sin_cos.shape[0]
    TERP_sin_cos = TERP_sin_cos.reshape(rows,-1)
    neighborhood_data = TERP_sin_cos
  else:
    if rows != TERP_sin_cos.shape[0]:
      logger1.error('Input (sine-cosine) dimension mismatch!')
      raise Exception()
    TERP_sin_cos = TERP_sin_cos.reshape(rows,-1)
    neighborhood_data = np.column_stack((neighborhood_data, TERP_sin_cos))
  logger1.info('Input includes sine-cosine data')

if '-TERP_image' in sys.argv:
  TERP_image = np.load(sys.argv[sys.argv.index('-TERP_image') + 1])
  if rows != 'null':
    logger1.error('Cannot combine images with other data types!')
    raise Exception()
  rows = TERP_image.shape[0]
  TERP_image = TERP_image.reshape(rows,-1)
  distance_metric = 'cosine'
  neighborhood_data = TERP_image
  logger1.info('Input contains image data')
else:
  distance_metric = 'euclidean'

if '-pred_proba' in sys.argv:
  pred_proba = np.load(sys.argv[sys.argv.index('-pred_proba') + 1])
  if pred_proba.shape[0] != rows:
    logger1.error('TERP input and  prediction probability dimension mismatch!')
    raise Exception()
  pred_proba = pred_proba.reshape(rows,-1)
else:
  logger1.error('Missing pred_proba data!')
  raise Exception()

if '-iterations' in sys.argv:
  iterations = int(sys.argv[sys.argv.index('-iterations') + 1])
  logger1.info('iterations :: ' + str(iterations))
else:
  iterations = 100000
  logger1.warning('iterations (number of monte carlo steps) not provided, defaulting to ::' + str(iterations))


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

if '-k_max' in sys.argv:
  k_max = int(sys.argv[sys.argv.index('-k_max') + 1])
  logger1.info('k_max :: ' + str(k_max))
  if k_max < 1 or k_max > neighborhood_data.shape[1]:
    logger1.error('Invalid k_max!')
    raise Exception
else:
  k_max = 5
  logger1.warning('k_max not prvided, defaulting to :: 5')

def similarity_kernel(data, kernel_width):
  distances = met.pairwise_distances(data,data[0].reshape(1, -1),metric=distance_metric).ravel()
  return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

if '--euclidean' in sys.argv:
  weights = similarity_kernel(neighborhood_data, 0.75*np.sqrt(neighborhood_data.shape[1]))
else:
  threshold, upper, lower = 0.5, 1, 0
  target_binarized = np.where(target>threshold, upper, lower)

  clf = lda()
  clf.fit(neighborhood_data,target_binarized)
  projected_data = clf.transform(neighborhood_data)
  weights = similarity_kernel(projected_data, 1)

if '-selected_features' in sys.argv:
  selected_features = np.load(sys.argv[sys.argv.index('-selected_features') + 1])
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

if k_max > len(selected_features):
  k_max = len(selected_features)
  print("k_max reselected to: ", k_max)
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
    asdasd = np.absolute(labels - (data@parameters[:-1]).reshape(-1,1))**2
    TERP_SGD_unfaithfulness.append(np.sum(asdasd))
  best_model = np.argsort(TERP_SGD_unfaithfulness)[0]
  best_parameters_master.append(TERP_SGD_parameters[best_model])
  best_unfaithfulness_master.append(TERP_SGD_unfaithfulness[best_model])

  asdasd = (data@TERP_SGD_parameters[best_model][:-1])
  corr_coef_master.append(np.corrcoef(labels.flatten(),asdasd)[0,1])
  print('Unfaithfulness: ', TERP_SGD_unfaithfulness[best_model])
  print('Relevant features: ', selected_features[np.nonzero(TERP_SGD_parameters[best_model][:-1])[0]])
best_parameters_master = []
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


np.save(results_directory + '/weights.npy', weights)
np.save(results_directory + '/coefficients.npy', np.array(best_parameters_master))
np.save(results_directory + '/unfaithfulness.npy', np.array(best_unfaithfulness_master))
np.save(results_directory + '/corr_coef.npy', np.array(corr_coef_master))

####### S
scan = np.linspace(0,((best_unfaithfulness_master[0]-best_unfaithfulness_master[1])/np.log(2)),1000)
S = np.log(k_array)
def term(alpha,S, U):
  return alpha*np.array(S) + U
min_sol = []
for i in scan:
  min_sol.append(np.argmin(term(i,S,best_unfaithfulness_master)))

counts = np.bincount(min_sol)
print(100*'-')
print('most relevant features obtained at k = ', np.argmax(counts)+1, ' model!')
######
endtime = time.time()
monte_carlo_time = endtime - starttime
print(100*'-')
logger2.info('computation time :: ' + str(int(monte_carlo_time/60)) + ' min ' + "{:.3f}".format(monte_carlo_time%60) + ' sec...')
print(100*'-')
