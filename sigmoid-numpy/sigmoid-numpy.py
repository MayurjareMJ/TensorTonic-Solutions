import numpy as np

def sigmoid(x):
   
    x = np.array(x, dtype=float)   # Convert input to NumPy array
    return 1 / (1 + np.exp(-x))