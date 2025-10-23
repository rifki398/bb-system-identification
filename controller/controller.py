import numpy as np
from scipy.signal import place_poles
    
def pole_placement(A,B):
    # Pole placement / full-state feedback controller
    desired_poles = np.array([-3, -3.5, -4, -4.5])

    K = place_poles(A, B, desired_poles).gain_matrix
    return K