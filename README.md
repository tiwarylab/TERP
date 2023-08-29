# TERP
Thermodynamically Explainable Representations of AI and other black-box Paradigms


TERP is a post-hoc interpretation scheme for explaining black-box AI predictions. TERP works by constructing a linear, local interpretable model that approximates the black-box in the vicinity of the instance being explained. TERP determines the accuracy-interpretability trade-off by introducing interpretation entropy.

# Usage
Getting TERP interpretation involves the following steps:

1. Feature selection: Generate neighborhood using TERP_neighborhood_generator.py
```
!python TERP_neighborhood_generator.py -seed 1 --progress_bar -input_numeric input_data.npy -num_samples 5000 -index 102975

-seed # random seed
-input_numeric # location of a numpy array with a representative distribution of the black-box model training data e.g, the training data itself
-num_samples # size of the generated neighborhood
-index # row index of the input (e.g, input_numeric) file whose prediction needs to be explained
```
2. obtain black-box model prediction by passing generated neighborhood saved at DATA/make_prediction_numeric.npy and save the predicted results in a numpy array. Rows of this array should represent datapoints and columns should represent different classes (e.g, neighborhood_state_probabilities.npy - see next step). Note a numpy array DATA/TERP_numeric.npy is created also created which will be used in the next step
3. Form a feature sub-space by identifying less/irrelevant features by constructing a linear model
```
!python TERP_optimizer_01.py -TERP_numeric DATA/TERP_numeric.npy -pred_proba DATA/neighborhood_state_probabilities.npy -cutoff 0.98

-TERP_numeric # Standardized neighborhood data location
-pred_proba # Prediction probabilities for different classes as obtained from the black-box model
-cutoff # Cutoff parameter to form a feature space. Higher values include more and more features. E.g, value of 1 will not discard any features (default 0.98)

Note: a numpy array selected_features.npy will be created at TERP_results/selected_features.npy
```
4. Generate a neighborhood by sampling the reduced feature space for improved interpretation
```
!python TERP_neighborhood_generator.py -seed 1 --progress_bar -input_numeric input_data.npy -num_samples 5000 -index 102975 -selected_features TERP_results/selected_features.npy

All the options are the same as in step 1. However, pass the additional -selected_features flag to analyze the sub-space only
```
5. obtain black-box model prediction by passing generated neighborhood saved at DATA_2/make_prediction_numeric.npy and save the predicted results in a numpy array. Rows of this array should represent datapoints and columns should represent different classes (e.g, neighborhood_state_probabilities.npy - see next step). Note a numpy array DATA_2/TERP_numeric.npy is created also created which will be used in the next step
6. Perform forward feature selection to obtain final result
```
!python TERP_optimizer_02.py -TERP_numeric DATA_2/TERP_numeric.npy -pred_proba DATA_2/neighborhood_state_probabilities.npy -selected_features TERP_results/selected_features.npy

Note: -pred_proba #prediction probabilites file for this step should be different from the one passed in step 3 because the neighborhood has been regenerated
```
