import numpy as np

def skew(v):
    '''Returns skew-symmetric matrix, which satisfies A^T = -A'''
    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])

def sensor_error_model(errors):
    '''
    Build sensor error model matrix and bias according to
    eq. (1) and (2) in the paper.
    '''
    KX, KY, KZ, NOX, NOY, NOZ, BX, BY, BZ = errors
    M = np.array([[1.0 + KX, NOY, NOZ], 
                  [0.0, 1.0 + KY, NOX], 
                  [0.0, 0.0, 1.0 + KZ]])
    b = np.array([BX, BY, BZ])
    
    return M, b

def misalignment(epsilon):
    '''
    Build rotation matrix from gyroscope coordinate frame
    to accelerometer coordinate frame according to eq. (5)
    '''
    return np.eye(3) + skew(epsilon)