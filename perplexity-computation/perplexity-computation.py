import numpy as np

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    N = len(prob_distributions)
    
    total_log_prob = 0.0
    
    for i in range(N):
        p_i = prob_distributions[i][actual_tokens[i]]  # probability of correct token
        total_log_prob += np.log(p_i)
    
    # Cross-entropy
    H = - total_log_prob / N
    
    # Perplexity
    return float(np.exp(H))