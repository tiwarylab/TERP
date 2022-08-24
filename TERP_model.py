"""
TERP: Thermodynamically Explainable Representations of AI and other black-box Paradigms
Code maintained by Shams
"""
import numpy as np
import os
import sys
import sklearn.metrics as met
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import logging
from functools import partial
import time
import pandas as pd
from tqdm import tqdm


cores = multiprocessing.cpu_count()
results_directory = 'TERP_results'
os.makedirs(results_directory, exist_ok = True)
rows = 'null'
neighborhood_data = 'null'
epsilon = 0.0000001
metropolis_ratio = 0.5
metropolis_param_init = 1.0 #Initial metropolis parameter
metropolis_param_cores = np.zeros((cores))
metropolis_param_cores[:] = metropolis_param_init
log_every_steps = 500 # Do not make this too small!

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
logger2 = logging.getLogger('monte_carlo')
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

if '-low_dim_rep' in sys.argv:
  low_dim_rep = np.load(sys.argv[sys.argv.index('-low_dim_rep') + 1])
  if low_dim_rep.shape[0] != rows:
    logger1.error('TERP input and low dimensional representation dimension mismatch!')
    raise Exception
  low_dim_rep = low_dim_rep.reshape(rows,-1)
  distance_metric = 'euclidean' ## To make sure euclidean distance is chosen even for image data
  logger1.info('Using low dimensional representation for similarity kernel computation')
else:
  logger1.info('Using input data for similarity kernel computation')

if '-iterations' in sys.argv:
  iterations = int(sys.argv[sys.argv.index('-iterations') + 1])
  logger1.info('iterations :: ' + str(iterations))
else:
  iterations = 100000
  logger1.warning('iterations (number of monte carlo steps) not provided, defaulting to ::' + str(iterations))

if '--saveall' in sys.argv:
  saveall = 'y'
  saveall_dir = results_directory + '/saveall'
  os.makedirs(saveall_dir, exist_ok = True)
  logger1.info('each iteration results from all cores will be saved')
else:
  saveall = 'n'
  logger1.warning('saveall disabled :: only final reuslts will be saved!')

if '-explain_class' in sys.argv:
  explain_class = int(sys.argv[sys.argv.index('-explain_class') + 1])
  logger1.info("Toatal number of classes :: " + str(pred_proba.shape[1]))
  logger1.info('explain_class :: ' + str(explain_class))
  if explain_class not in [i for i in range(pred_proba.shape[1])]:
    logger1.error('Invalid -explain_class!')
    raise Exception()
else:
  explain_class = np.argmax(pred_proba[0,:])
  logger1.warning('explain_class not provided, defaulting to class with maximum predictiion probability :: ' + str(explain_class))

if '-kernel_grid_start' in sys.argv:
  kernel_grid_start = float(sys.argv[sys.argv.index('-kernel_grid_start') + 1])
else:
  kernel_grid_start = 0.001

if '-kernel_grid_end' in sys.argv:
  kernel_grid_end = float(sys.argv[sys.argv.index('-kernel_grid_end') + 1])
else:
  kernel_grid_end = 5

if '-kernel_num_grids' in sys.argv:
  kernel_num_grids = int(sys.argv[sys.argv.index('-kernel_num_grids') + 1])
else:
  kernel_num_grids = 20

logger1.info('kernel_grid_start, kernel_grid_end, kernel_num_grids :: ' + str(kernel_grid_start) + ', ' + str(kernel_grid_end) + ', ' + str(kernel_num_grids))

if '-k_max' in sys.argv:
  k_max = int(sys.argv[sys.argv.index('-k_max') + 1])
  logger1.info('k_max :: ' + str(k_max))
  if k_max < 1 or k_max > neighborhood_data.shape[1]:
    logger1.error('Invalid k_max!')
    raise Exception
else:
  k_max = 5
  logger1.warning('k_max not prvided, defaulting to :: 5')

def distances_func(data):
  return met.pairwise_distances(data,data[0].reshape(1, -1),metric=distance_metric).ravel()

def similarity_kernel(distances, kernel_width):
    return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

def metropolis_ratio_update(metropolis_parameter, current_ratio):
    if current_ratio>2*metropolis_ratio:
      updated_param = metropolis_parameter/2
    else:
      updated_param = metropolis_parameter + metropolis_parameter*((metropolis_ratio-current_ratio)/metropolis_ratio)
    return updated_param

def optimized_sim_kernel():
  ## This function obtains optimal kernel width that maximizes standard deviation of the similarity kernel distribution
  if '-low_dim_rep' in sys.argv:
    low_dim_rep_TERP = np.zeros((low_dim_rep.shape[0], low_dim_rep.shape[1]))
    for i in range(low_dim_rep.shape[1]):
      low_dim_rep_TERP[:,i] = (low_dim_rep[:,i]-np.mean(low_dim_rep[:,i]))/np.std(low_dim_rep[:,i])
    distances_1 = distances_func(low_dim_rep_TERP)
    distances_2 = distances_func(neighborhood_data)
    distances = (distances_1/low_dim_rep_TERP.shape[1]) + (distances_2/neighborhood_data.shape[1])
  else:
    distances = distances_func(neighborhood_data)/neighborhood_data.shape[1]
  kernel_grid = np.linspace(kernel_grid_start, kernel_grid_end, kernel_num_grids)
  kernel_dist_std = []
  for i in range(kernel_grid.shape[0]):
    weights = similarity_kernel(distances, kernel_grid[i])
    kernel_dist_std.append(np.std(weights))
  optimal_kernel_width_index = np.argmax(kernel_dist_std)
  weights = similarity_kernel(distances, kernel_grid[optimal_kernel_width_index])
  fig, ax = plt.subplots(figsize = (5,5))
  ax.hist(weights, bins = 10)
  ax.set_ylabel('count')
  ax.set_xlabel('similarity')
  ax.set_xlim(0,1)
  fig.tight_layout()
  fig.savefig("TERP_results/similarity_distribution.png",cpi=300,bbox_inches='tight')
  logger1.info("optimal kernel width :: " + "{:.6f}".format(kernel_grid[optimal_kernel_width_index]))
  return weights

def weighted_local_model_value(data, weights, coefficient_array, intercept):
  local_model_mat = data*coefficient_array
  matrix = np.sum(local_model_mat, axis = 1) + intercept
  weighted_matrix = np.multiply(np.sqrt(weights), matrix)
  return weighted_matrix

def monte_carlo(core_id, k, N, neighborhood_data, predict_proba, weights, metropolis_param, iterations, weighted_obscure_model_pred, coefficient_array, intercept_array):
  np.random.seed(core_id)
  tot_weight = np.sum(weights)

  inherited_nonzero = np.nonzero(coefficient_array)[0]
  zeros_to_perturb =  k - inherited_nonzero.shape[0]
  inherited_zero = np.where(coefficient_array == 0)[0]
  needed_random_draws = np.random.choice(inherited_zero, size = zeros_to_perturb, replace = False)

  for j in range(needed_random_draws.shape[0]):
    coefficient_array[needed_random_draws[j]] = epsilon

  coefficient_array_binarized = np.zeros((N))
  coefficient_array_binarized[np.nonzero(coefficient_array)[0]] = 1

  accepted_count = 0
  metropolis_accepted_count = 0
  rejected_count = 0

  weighted_local_model_pred = weighted_local_model_value(neighborhood_data, weights, coefficient_array, intercept_array)
  weighted_residual_matrix = (weighted_obscure_model_pred-weighted_local_model_pred)**2
  unfaithfulness = np.sum(weighted_residual_matrix)

  best_unfaithfulness = unfaithfulness
  best_coefficients_array = coefficient_array
  best_intercept_array = intercept_array

  if saveall == 'y':
    data_best_unfaithfulness = np.zeros((log_every_steps))
    data_unfaithfulness = np.zeros((log_every_steps))
    data_coefficients = np.zeros((log_every_steps,N))
    data_intercept = np.zeros((log_every_steps))

    data_best_unfaithfulness[0] = best_unfaithfulness
    data_unfaithfulness[0] = unfaithfulness
    data_coefficients[0,:] = coefficient_array
    data_intercept[0] = intercept_array

  current_step = 1
  for current_step in range(1,iterations+1) :
    coefficient_array_current = np.zeros((N))
    R = np.random.uniform(low=0.0, high = 1.0)

    q = np.random.uniform(low = -0.5, high = 0.5, size = N+1)/10
    
    coefficient_array_current = coefficient_array + coefficient_array_binarized*q[:N]
    intercept_array_current = intercept_array + q[N]
    ## Swap coefficients using tuple unpacking
    if k>0:
      p = np.random.choice(np.arange(N),size = 1, replace = True)
      p2 = np.random.choice(np.where(coefficient_array_binarized == 1)[0],size = 1, replace = True)
      coefficient_array_current[p2], coefficient_array_current[p] = coefficient_array_current[p], coefficient_array_current[p2]

    updated_weighted_local_model_pred = weighted_local_model_value(neighborhood_data, weights, coefficient_array_current, intercept_array_current)
    updated_weighted_residual_matrix = (weighted_obscure_model_pred-updated_weighted_local_model_pred)**2

    unfaithfulness_current = np.sum(updated_weighted_residual_matrix)

    if unfaithfulness_current < unfaithfulness:
      ## Keep the updates
      coefficient_array = coefficient_array_current
      intercept_array = intercept_array_current
      unfaithfulness = unfaithfulness_current
      if k>0:
        coefficient_array_binarized[p2], coefficient_array_binarized[p] = coefficient_array_binarized[p], coefficient_array_binarized[p2]
      accepted_count += 1
    elif np.exp(-(unfaithfulness_current-unfaithfulness)/(np.sqrt(k+1)*metropolis_param_cores[core_id])) > R:
      ## Keep the updates
      coefficient_array = coefficient_array_current
      intercept_array = intercept_array_current
      unfaithfulness = unfaithfulness_current
      if k>0:
        coefficient_array_binarized[p2], coefficient_array_binarized[p] = coefficient_array_binarized[p], coefficient_array_binarized[p2]
      metropolis_accepted_count += 1
    else:
      rejected_count += 1

    if unfaithfulness_current < best_unfaithfulness:
      best_unfaithfulness = unfaithfulness_current
      best_coefficients_array = coefficient_array_current
      best_intercept_array = intercept_array_current

    if current_step%log_every_steps ==0:
        if saveall == 'y':
          
          df_1 = pd.DataFrame(np.column_stack((data_best_unfaithfulness/np.sum(weights), data_unfaithfulness/np.sum(weights))), columns = ['best_unfaithfulness','unfaithfulness'])
          df_2 = pd.DataFrame(np.column_stack((data_coefficients, data_intercept)), columns = ['intercept'] + ['coeff_' + str(i) for i in range(N)])
          df_1.to_csv(saveall_dir + '/data1_k_' + str(k) + '_core_id_' + str(core_id) + '.csv', mode='w' if current_step == log_every_steps else 'a', header=True if current_step == log_every_steps else False, index = False)
          df_2.to_csv(saveall_dir + '/data2_k_' + str(k) + '_core_id_' + str(core_id) + '.csv', mode='w' if current_step == log_every_steps else 'a', header=True if current_step == log_every_steps else False, index = False)

        if rejected_count == 0:
          rejected_count = 1 ## To simplify r_ratio calculations
        if metropolis_accepted_count == 0:
          metropolis_accepted_count == 1 ## To simplify r_ratio calculations
        r_ratio = metropolis_accepted_count/rejected_count
        logger2.info('core_id :: ' + str(core_id) + ' , step :: ' + str(current_step) + ', rejection_ratio :: ' + "{:.2f}".format(r_ratio) + ', metropolis_parameter ::'+ "{:.3f}".format(metropolis_param_cores[core_id]))
        
        metropolis_param_cores[core_id] = metropolis_ratio_update(metropolis_param_cores[core_id], r_ratio)
        accepted_count = 0
        metropolis_accepted_count = 0
        rejected_count = 0
   
    if saveall == 'y':
      data_best_unfaithfulness[current_step%log_every_steps] = best_unfaithfulness
      data_unfaithfulness[current_step%log_every_steps] = unfaithfulness_current
      data_coefficients[current_step%log_every_steps] = coefficient_array
      data_intercept[current_step%log_every_steps] = intercept_array


  metropolis_param_cores[core_id] = metropolis_param_init ## Resetting metropolis parameter for k=k+1
  return best_coefficients_array, best_intercept_array, best_unfaithfulness/np.sum(weights) ## Saving final results for specific k and core id


def convergence_check(c_array_store):
  c_array_store_2 = c_array_store.reshape(c_array_store.shape[0],-1)
  highest_std = 0
  ret_val = 0
  for i in range(c_array_store_2.shape[1]):
    if np.count_nonzero(c_array_store_2 [:,i]) == 0:
      continue
    elif np.count_nonzero(c_array_store_2[:,i]) == c_array_store_2.shape[0]:
      if np.std(c_array_store_2[:,i]) > highest_std:
        highest_std = np.std(c_array_store_2[:,i])
        ret_val = np.std(c_array_store_2[:,i])/np.mean(c_array_store_2[:,i])
    else:
      return -1
  return ret_val


def unfaithfulness_calc(k, N, neighborhood_data, predict_proba, weights, metropolis_param, iterations, best_coefficients_master, best_intercept_master, best_unfaithfulness_master):
  weighted_obscure_model_pred = np.multiply(np.sqrt(weights), predict_proba)
  coefficient_array = np.zeros((N))

  c_array_store = np.zeros((cores,N))
  i_array_store = np.zeros((cores))
  b_u_array_store = np.zeros((cores))

  if k == 0:
    intercept_array = 1

  elif k > 0:
    coefficient_array[:] = best_coefficients_master[k-1][:]
    intercept_array = best_intercept_master[k-1]

  pool = Pool()
  monte_carlo_parallel = partial(monte_carlo, k=k, N=N, neighborhood_data=neighborhood_data, predict_proba=predict_proba, weights=weights, metropolis_param=metropolis_param, iterations=iterations, weighted_obscure_model_pred=weighted_obscure_model_pred, coefficient_array=coefficient_array, intercept_array=intercept_array)
  temp = pool.map(monte_carlo_parallel, range(cores))
  for i in range(cores):

    c_array_store[i,:] = temp[i][0]
    i_array_store[i] = temp[i][1]
    b_u_array_store[i] = temp[i][2]
  pool.close()
  pool.join()
  best_core_result = np.argmin(b_u_array_store)

  if k == 0:
    best_coefficients_master.append(c_array_store[best_core_result])
    best_intercept_master.append(i_array_store[best_core_result])
    best_unfaithfulness_master.append(b_u_array_store[best_core_result])

  elif (k>0) and (b_u_array_store[best_core_result]>best_unfaithfulness_master[k-1]):
    best_coefficients_master.append(best_coefficients_master[k-1])
    best_intercept_master.append(best_intercept_master[k-1])
    best_unfaithfulness_master.append(best_unfaithfulness_master[k-1])
  else:
    best_coefficients_master.append(c_array_store[best_core_result])
    best_intercept_master.append(i_array_store[best_core_result])
    best_unfaithfulness_master.append(b_u_array_store[best_core_result])
  return convergence_check(c_array_store)  


best_coefficients_master = []
best_intercept_master = []
best_unfaithfulness_master = []

N = neighborhood_data.shape[1]
k_array = np.arange(k_max + 1)


weights = optimized_sim_kernel()
logger1.info('Similarity computation complete...')
print(100*'-')
logger2.info('Starting monte carlo using (' + str (cores) + ') detected processor cores...')
print(100*'-')
starttime = time.time()
convergence_lst = []
for k in k_array:
  print(100*'-')
  logger2.info('Scanning models for k :: ' + str(k))
  print(100*'-')
  convergence_val = unfaithfulness_calc(k, N, neighborhood_data, pred_proba[:,explain_class], weights, metropolis_param_init, iterations, best_coefficients_master, best_intercept_master, best_unfaithfulness_master)
  convergence_lst.append(convergence_val)

np.save(results_directory + '/coefficients.npy', np.array(best_coefficients_master))
np.save(results_directory + '/intercept.npy', np.array(best_intercept_master))
np.save(results_directory + '/unfaithfulness.npy', np.array(best_unfaithfulness_master))
np.save(results_directory + '/convergence.npy', np.array(convergence_lst))

endtime = time.time()
monte_carlo_time = endtime - starttime
print(100*'-')
logger2.info('computation time :: ' + str(int(monte_carlo_time/60)) + ' min ' + "{:.3f}".format(monte_carlo_time%60) + ' sec...')
print(100*'-')
