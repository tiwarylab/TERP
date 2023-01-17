"""
TERP: Thermodynamically Explainable Representations of AI and other black-box Paradigms
"""
import numpy as np
import os
import sys
import copy

def generate_neighborhood():

    save_directory = 'DATA'
    os.makedirs(save_directory, exist_ok = True)
    rows = 'null'
    # Be cautious if need to change periodicity. Some parts are hard-coded.
    period_high = np.pi
    period_low = -np.pi

    if '-input_categorical' in sys.argv:
      categorical = np.load(sys.argv[sys.argv.index('-input_categorical') + 1])
      rows = categorical.shape[0]
      categorical = categorical.reshape(rows,-1)
      if '-selected_features' in sys.argv:
        selected_features = np.load(sys.argv[sys.argv.index('-selected_features') + 1])
        input_categorical = categorical[:,selected_features]
      else:
        input_categorical = copy.deepcopy(categorical)
      print(">>> Categorical data provided...")

    if '-input_numeric' in sys.argv:
      numeric = np.load(sys.argv[sys.argv.index('-input_numeric') + 1])
      print(">>> Numerical data provided...")
      if rows == 'null':
        rows = numeric.shape[0]
        numeric = numeric.reshape(rows,-1)
      else:
        assert rows == numeric.shape[0], "Input data dimension mismatch..."
        numeric = numeric.reshape(rows,-1)
      if '-selected_features' in sys.argv:
        selected_features = np.load(sys.argv[sys.argv.index('-selected_features') + 1])
        input_numeric = numeric[:,selected_features]
      else:
        input_numeric = copy.deepcopy(numeric)

    if '-input_periodic' in sys.argv:
      import scipy.stats as sst
      periodic = np.load(sys.argv[sys.argv.index('-input_periodic') + 1])
      assert np.all(periodic<=np.pi) and np.all(periodic>-np.pi), 'Provide periodic data in domain (-pi,pi]...'
      print(">>> Periodic data provided...")
      if rows == 'null':
        rows = periodic.shape[0]
        periodic = periodic.reshape(rows,-1)
      else:
        assert rows == periodic.shape[0], "Input data dimension mismatch..."
        periodic = periodic.reshape(rows,-1)
      if '-selected_features' in sys.argv:
        selected_features = np.load(sys.argv[sys.argv.index('-selected_features') + 1])
        input_periodic = periodic[:,selected_features]
      else:
        input_periodic = copy.deepcopy(periodic)

    if '-input_sin' in sys.argv:
      sin = np.load(sys.argv[sys.argv.index('-input_sin') + 1])
      assert np.all(sin>=-1) and np.all(sin<=1), 'Provide sine data in domain [-1,1]'
      if rows == 'null':
        rows = sin.shape[0]
        sin = sin.reshape(rows,-1)
      else:
        assert rows == sin.shape[0], "Input data dimension mismatch..."
        sin = sin.reshape(rows,-1)
      if '-selected_features' in sys.argv:
        selected_features = np.load(sys.argv[sys.argv.index('-selected_features') + 1])
        input_sin = sin[:,selected_features]
      else:
        input_sin = copy.deepcopy(sin)

    if '-input_cos' in sys.argv:
      cos = np.load(sys.argv[sys.argv.index('-input_cos') + 1])
      assert np.all(cos>=-1) and np.all(cos<=1), 'Provide cosine data in domain [-1,1]'
      if rows == 'null':
        rows = cos.shape[0]
        cos = cos.reshape(rows,-1)
      else:
        assert rows == cos.shape[0], "Input data dimension mismatch..."
        cos = cos.reshape(rows,-1)
      if '-selected_features' in sys.argv:
        selected_features = np.load(sys.argv[sys.argv.index('-selected_features') + 1])
        input_cos = cos[:,selected_features]
      else:
        input_cos = copy.deepcopy(cos)

    if '-input_sin' in sys.argv or '-input_cos' in sys.argv:
      import scipy.stats as sst

      assert sin.shape == cos.shape, "Sin-cos data dimension mismatch..."
      print(">>> Sin-cos data provided...")

    if '-input_image' in sys.argv:
      from PIL import Image
      from skimage.segmentation import slic, mark_boundaries, quickshift
      import matplotlib.pyplot as plt

      input_image = Image.open(sys.argv[sys.argv.index('-input_image') + 1])
      print(">>> Image data provided...")
      assert rows == 'null', 'Cannot combine images with other data types!!'

    if '-image_segments' in sys.argv:
      image_segments = int(sys.argv[sys.argv.index('-image_segments') + 1])
    else:
      image_segments = 50

    if '-image_compactness' in sys.argv:
      image_compactness = int(sys.argv[sys.argv.index('-image_compactness') + 1])
    else:
      image_compactness = image_segments*2

    if '-index' in sys.argv:
      index = int(sys.argv[sys.argv.index('-index') + 1])
    else:
      index = 0
      if '-input_image' not in sys.argv:
        print(">>> No index provided, explaining first data point...")
    
    if '-seed' in sys.argv:
      seed = int(sys.argv[sys.argv.index('-seed') + 1])
      np.random.seed(seed)
    else:
      np.random.seed(0)
      print(">>> No random seed provided, using (0)...")

    if '-num_samples' in sys.argv:
      num_samples = int(sys.argv[sys.argv.index('-num_samples') + 1])
    else:
      num_samples = 2000
      print('>>> Neighborhood size not provided, using (2000)...')

    if '--progress_bar' in sys.argv:
        from tqdm import tqdm
        progress_bar = 'y'
    else:
        loop_iterator = range(num_samples)
        progress_bar = 'n'

    #### Generate data
    if rows != 'null' or '-input_image' in sys.argv:
      print(">>> Generating data...")

    if '-input_categorical' in sys.argv:
      make_prediction_categorical = np.zeros((num_samples, input_categorical.shape[1]))
      TERP_categorical = np.zeros((num_samples, input_categorical.shape[1]))

      perturb_categorical = np.random.randint(0, 2, num_samples * input_categorical.shape[1]).reshape((num_samples, input_categorical.shape[1]))
      perturb_categorical[0,:] = 1

      if progress_bar == 'y':   
        loop_iterator = tqdm(range(num_samples))

      for i in loop_iterator:
        for j in range(input_categorical.shape[1]):
          if perturb_categorical[i,j] == 1:
            make_prediction_categorical[i,j] = input_categorical[index,j]
            TERP_categorical[i,j] = 1
          elif perturb_categorical[i,j] == 0:
            make_prediction_categorical[i,j] = np.random.choice(input_categorical[:,j])
            if make_prediction_categorical[i,j] == input_categorical[index,j]:
              TERP_categorical[i,j] = 1

      if '-selected_features' in sys.argv:
        temp = np.zeros((make_prediction_categorical.shape[0],categorical.shape[1]))
        temp = temp + categorical[index,:].reshape(-1,1)
        temp[:,selected_features] = make_prediction_categorical
        make_prediction_categorical = copy.deepcopy(temp)
      
      np.save(save_directory + '/make_prediction_categorical.npy', make_prediction_categorical)            
      np.save(save_directory + '/TERP_categorical.npy', TERP_categorical)    


    if  '-input_numeric' in sys.argv:
      std_numeric = []
      for i in range(input_numeric.shape[1]):
        std_numeric.append(np.std(input_numeric[:,i]))

      make_prediction_numeric = np.zeros((num_samples, input_numeric.shape[1]))
      TERP_numeric = np.zeros((num_samples, input_numeric.shape[1]))

      perturb_numeric = np.random.randint(0, 2, num_samples * input_numeric.shape[1]).reshape((num_samples, input_numeric.shape[1]))
      perturb_numeric[0,:] = 1

      if progress_bar == 'y':   
        loop_iterator = tqdm(range(num_samples))
        
      for i in loop_iterator:
        for j in range(input_numeric.shape[1]):
          if perturb_numeric[i,j] == 1:
            make_prediction_numeric[i,j] = input_numeric[index,j]
          elif perturb_numeric[i,j] == 0:
            rand_data = np.random.normal(0, 1)
            make_prediction_numeric[i,j] = input_numeric[index,j] + std_numeric[j]*rand_data
            TERP_numeric[i,j] = rand_data

      if '-selected_features' in sys.argv:
        temp = np.zeros((make_prediction_numeric.shape[0],numeric.shape[1]))
        temp = temp + numeric[index,:]
        temp[:,selected_features] = make_prediction_numeric
        make_prediction_numeric = copy.deepcopy(temp)
        
        np.save(save_directory + '/make_prediction_numeric_2.npy', make_prediction_numeric)            
        np.save(save_directory + '/TERP_numeric_2.npy', TERP_numeric)    

      else:
        np.save(save_directory + '/make_prediction_numeric.npy', make_prediction_numeric)            
        np.save(save_directory + '/TERP_numeric.npy', TERP_numeric)    


    if  '-input_periodic' in sys.argv:
      std_periodic = []
      for i in range(input_periodic.shape[1]):
        std_periodic.append(sst.circstd(input_periodic[:,i], high = period_high, low = period_low))

      make_prediction_periodic = np.zeros((num_samples, input_periodic.shape[1]))
      TERP_periodic = np.zeros((num_samples, input_periodic.shape[1]))

      perturb_periodic = np.random.randint(0, 2, num_samples * input_periodic.shape[1]).reshape((num_samples, input_periodic.shape[1]))
      perturb_periodic[0,:] = 1

      if progress_bar == 'y':   
        loop_iterator = tqdm(range(num_samples))

      for i in loop_iterator:
        for j in range(input_periodic.shape[1]):
          if perturb_periodic[i,j] == 1:
            make_prediction_periodic[i,j] = input_periodic[index,j]
          elif perturb_periodic[i,j] == 0:
            rand_data = np.random.normal(0, 1)
            make_prediction_periodic[i,j] = input_periodic[index,j] + std_periodic[j]*rand_data
            TERP_periodic[i,j] = rand_data
            if make_prediction_periodic[i,j] < period_low or make_prediction_periodic[i,j] > period_high:
              make_prediction_periodic[i,j] = np.arctan2(np.sin(make_prediction_periodic[i,j]), np.cos(make_prediction_periodic[i,j]))

      if '-selected_features' in sys.argv:
        temp = np.zeros((make_prediction_periodic.shape[0],periodic.shape[1]))
        temp = temp + periodic[index,:]
        temp[:,selected_features] = make_prediction_periodic
        make_prediction_periodic = copy.deepcopy(temp)

      np.save(save_directory + '/make_prediction_periodic.npy', make_prediction_periodic)            
      np.save(save_directory + '/TERP_periodic.npy', TERP_periodic)    


    if  '-input_sin' in sys.argv:
      std_sin_cos = []
      input_sin_cos = np.zeros((input_sin.shape[0], input_sin.shape[1])).reshape(rows,-1)
      for i in range(input_sin.shape[1]):
        input_sin_cos[:,i] = np.arctan2(input_sin[:,i], input_cos[:,i])
        std_sin_cos.append(sst.circstd(input_sin_cos[:,i], high = period_high, low = period_low))

      make_prediction_sin = np.zeros((num_samples, input_sin_cos.shape[1]))
      make_prediction_cos = np.zeros((num_samples, input_sin_cos.shape[1]))
      TERP_sin_cos = np.zeros((num_samples, input_sin_cos.shape[1]))

      perturb_sin_cos = np.random.randint(0, 2, num_samples * input_sin_cos.shape[1]).reshape((num_samples, input_sin_cos.shape[1]))
      perturb_sin_cos[0,:] = 1

      if progress_bar == 'y':   
        loop_iterator = tqdm(range(num_samples))

      for i in loop_iterator:
        for j in range(input_sin_cos.shape[1]):
          if perturb_sin_cos[i,j] == 1:
            make_prediction_sin[i,j] = np.sin(input_sin_cos[index,j])
            make_prediction_cos[i,j] = np.cos(input_sin_cos[index,j])

          elif perturb_sin_cos[i,j] == 0:
            rand_data = np.random.normal(0, 1)
            make_prediction_sin[i,j] = np.sin(input_sin_cos[index,j] + std_sin_cos[j]*rand_data)
            make_prediction_cos[i,j] = np.cos(input_sin_cos[index,j] + std_sin_cos[j]*rand_data)
            TERP_sin_cos[i,j] = rand_data

      if '-selected_features' in sys.argv:
        temp = np.zeros((make_prediction_sin.shape[0],sin.shape[1]))
        temp = temp + sin[index,:]
        temp[:,selected_features] = make_prediction_sin
        make_prediction_sin = copy.deepcopy(temp)

        temp = np.zeros((make_prediction_cos.shape[0],cos.shape[1]))
        temp = temp + cos[index,:]
        temp[:,selected_features] = make_prediction_cos
        make_prediction_cos = copy.deepcopy(temp)

      np.save(save_directory + '/make_prediction_sin.npy', make_prediction_sin)  
      np.save(save_directory + '/make_prediction_cos.npy', make_prediction_cos)          
      np.save(save_directory + '/TERP_sin_cos.npy', TERP_sin_cos)   
    
    if '-input_image' in sys.argv:
      os.makedirs(save_directory + '/perturbed_images', exist_ok = True)

      segments = slic(input_image,n_segments=image_segments,compactness=image_compactness)
      fig,ax = plt.subplots(figsize=(8, 8))
      ax.imshow(mark_boundaries(input_image, segments))
      fig.savefig(save_directory + '/superpixels.png',bbox_inches='tight',dpi=300)

      rgb_image = np.array(input_image.getdata()).reshape((input_image.size[1], input_image.size[0],3))
      fudged_image = rgb_image.copy()

      n_features = np.unique(segments).shape[0]
      temp = np.zeros((num_samples, n_features))
      if '-selected_features' in sys.argv:
        selected_features = np.load(sys.argv[sys.argv.index('-selected_features') + 1])
      data = np.random.randint(0, 2, num_samples * selected_features.shape[0]).reshape((num_samples, selected_features.shape[0]))
      temp[:,selected_features] = data
      data = copy.deepcopy(temp)
      labels = []
      data[0, :] = 1
      for x in np.unique(segments):
          fudged_image[segments == x] = (
              np.mean(rgb_image[segments == x][:, 0]),
              np.mean(rgb_image[segments == x][:, 1]),
              np.mean(rgb_image[segments == x][:, 2]))
      
      counter = 0
      if progress_bar == 'y':
        rows = tqdm(data)
      else:
        rows = data
      for row in rows:
          temp = copy.deepcopy(rgb_image)
          zeros = np.where(row == 0)[0]
          mask = np.zeros(segments.shape).astype(bool)
          for z in zeros:
              mask[segments == z] = True
          temp[mask] = fudged_image[mask]
          Image.fromarray(np.uint8(temp)).convert('RGB').save(save_directory + '/perturbed_images/image_' + format(counter,'06d') + '.png')
          counter += 1

      np.save(save_directory + '/TERP_image.npy', data)
      np.save(save_directory + '/image_segments.npy', segments)

    if type(rows) != 'str':
      print('>>> Data generation complete!')
    else:
      print('>>> Incorrect command. No data generated!')

if __name__ == '__main__':
    generate_neighborhood()
    
