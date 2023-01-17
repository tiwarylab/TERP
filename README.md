# TERP
Thermodynamically Explainable Representations of AI and other black-box Paradigms


TERP is a post-hoc interpretation scheme for explaining black-box AI predictions. TERP works by constructing a linear, local interpretable model that approximates the black-box in the vicinity of the instance being explained.

# Usage
Getting TERP interpretation is a 4 step process:

1. Feature selection: Generate neighborhood using TERP_gen_data.py
2. Form a feature sub-space by identifying less/irrelevant features by constructing a linear model
3. Generate a neighborhood by sampling the reduced feature space
4. Perform forward feature selection to obtain final result
