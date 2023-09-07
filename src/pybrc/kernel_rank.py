import numpy as np
import scipy.linalg as sla

def kernel_rank(X, cutoff=0.99):
    #define input signal
    #y = 2*rand(num_timesteps,n_input_units)-1;
    #input_sequence = repmat(y[:,1],1,n_input_units);
    # rescale for each reservoir
    #[input_sequence] = featureNormailse(input_sequence,config.preprocess);

    #kernel matrix - pick 'to' at halfway point
    #M = config.assessFcn(individual,input_sequence,config);

    #catch errors
    #M(isnan(M)) = 0;
    #M(isinf(M)) = 0;

    #% Kernal Quality
    singular_values = sla.svd(X, compute_uv=False)

    full_sum = singular_values.sum()
    cumulative_sum = np.cumsum(singular_values)
    # For line below, searchsorted finds the number of singular values (in decreasing order)
    # that the summation is (cutoff)% of the total sum.
    return np.searchsorted(cumulative_sum, full_sum * cutoff) + 1
