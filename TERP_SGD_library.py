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


if '-cutoff' in sys.argv:
  cutoff = float(sys.argv[sys.argv.index('-cutoff') + 1])
else:
  cutoff = 0.9

if '--euclidean' in sys.argv:
  weights = similarity_kernel(neighborhood_data, 0.75*np.sqrt(neighborhood_data.shape[1]))
else:
  threshold, upper, lower = 0.5, 1, 0
  target_binarized = np.where(target>threshold, upper, lower)

  clf = lda()
  clf.fit(neighborhood_data,target_binarized)
  projected_data = clf.transform(neighborhood_data)
  weights = similarity_kernel(projected_data, 1)

predict_proba = pred_proba[:,explain_class]
data = neighborhood_data*(weights**0.5).reshape(-1,1)
labels = target.reshape(-1,1)*(weights.reshape(-1,1)**0.5)

def SGDreg(predict_proba, data, labels):
  clf = Ridge(alpha=1.0, random_state = 10, solver = 'sag')
  clf.fit(data,labels.ravel())
  coefficients = clf.coef_
  intercept = clf.intercept_
  return coefficients, intercept

def selection(coefficients, threshold):
  coefficients_abs = np.absolute(coefficients)
  np.save('abs.npy',coefficients_abs)
  selected_features = []
  coverage = 0
  for i in range(coefficients_abs.shape[0]):
    coverage = coverage+np.sort(coefficients_abs)[::-1][i]/np.sum(coefficients_abs)
    selected_features.append(np.argsort(coefficients_abs)[::-1][i])
    if coverage>threshold:
      break
  return selected_features

coefficients_selection, intercept_selection = SGDreg(predict_proba, data, labels)
selected_features = selection(coefficients_selection, cutoff)
print('Selected the following ' + str(len(selected_features)) + ' features!')
print(selected_features)
np.save(results_directory + '/selected_features.npy', selected_features)
np.save(results_directory + '/weights_step_01.npy', weights)
