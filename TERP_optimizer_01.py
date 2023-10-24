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

results_directory = 'TERP_results'
os.makedirs(results_directory, exist_ok = True)
rows = 'null'
neighborhood_data = 'null'
############################################
# Set up logger
fmt = '%(asctime)s %(name)-15s %(levelname)-8s %(message)s'
datefmt='%m-%d-%y %H:%M:%S'
logging.basicConfig(level=logging.INFO,format=fmt,datefmt=datefmt,filename=results_directory+'/TERP_1.log',filemode='w')
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
  cutoff = int(sys.argv[sys.argv.index('-cutoff') + 1])
  logger1.info("Provided cutoff :: " + str(cutoff))
else:
  cutoff = 25
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
  weights = similarity_kernel(projected_data.reshape(-1,1), 1)
  logger1.info("No distance flag provided. Performing 1-d LDA projection to compute similarity measure!")


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
  selected_features = []
  coverage = 0
  for i in range(coefficients_abs.shape[0]):
    if i==threshold:
      break
    coverage = coverage+np.sort(coefficients_abs)[::-1][i]/np.sum(coefficients_abs)
    selected_features.append(np.argsort(coefficients_abs)[::-1][i])
  logger1.warning('Top ' + str(threshold) + ' features selected with weight coverage :: ' + str(coverage) + '!!')
  return selected_features

coefficients_selection, intercept_selection = SGDreg(predict_proba, data, labels)
coefficients_selection = coefficients_selection/np.sum(np.absolute(coefficients_selection))
selected_features = selection(coefficients_selection, cutoff)
logger1.info('Selected the following ' + str(len(selected_features)) + ' out of (' + str(TERP_input.shape[1]) + ') features to form a feature sub-space ::')
logger1.info(selected_features)

with open(results_directory + '/selected_features.npy', "wb") as fp:
  pickle.dump([selected_features, data.shape[1]], fp)

np.save(results_directory + '/similarity_feature_selection.npy', weights)
np.save(results_directory + '/coefficients_feature_selection.npy', coefficients_selection)
np.save(results_directory + '/intercept_feature_selection.npy', intercept_selection)
