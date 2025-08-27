import numpy as np

def generate_dataset(model, N=10000):
    '''
    **Input** -------------------- \n
    `N`: data set size \n
    `model`: Real plant model \n
    **Output** ------------------- \n
    `X`: input data set for NN \n
    `Y`: output data set for NN
    '''
    np.random.seed(0)
    X = [] # [x1 x2 x3 x4 u]
    Y = [] # [x1dot x2dot x3dot x4dot]

    for _ in range(N):
        x = np.random.uniform(low=[-1, -1, -2, -2], 
                              high=[1, 1, 2, 2])
        u = np.random.uniform(-1.0, 1.0)
        xdot = model.f(x, u)
        X.append(np.concatenate((x, [u])))
        Y.append(xdot)
        
    return np.array(X), np.array(Y)