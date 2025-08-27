import numpy as np

def test(x,u,x_nn,real_model,id_model,Ts=0.1):
    
    xdot_true = real_model.f(x, u)
    xdot_pred = id_model(x_nn).detach().numpy()
    
    # Forward Euler method
    x += Ts * xdot_true
    x_nn += Ts * np.concatenate((xdot_pred, [0]))

    return x,x_nn.numpy()