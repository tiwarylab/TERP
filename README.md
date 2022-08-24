# TERP
Thermodynamically Explainable Representations of AI and other black-box Paradigms


TERP is a post-hoc interpretation scheme for explaining black-box AI predictions. TERP works by constructing a linear, local interpretable model that approximates the black-box in the vicinity of the instance being explained.

# Usage
First process data using TERP_gen_data.py.
Example:
```
!python TERP_gen_data.py -seed 150 --progress_bar -input_categorical categorical.npy -num_samples 5000 -index -3 -input_numeric numercal.npy
```
Obtain black-box predictions using generated neighborhood saved in 'DATA' subdirectory
Afterwards implement forward feature selection Monte Carlo using TERP_model.py.
Example:
```
!python TERP_model.py -TERP_categorical DATA/TERP_categorical.npy -TERP_numeric DATA/TERP_numeric.npy -pred_proba explain_pred.npy -iterations 20000 -k_max 8 --saveall -explain_class 1
```
