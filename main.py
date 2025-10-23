import numpy as np
import torch
from lib import *
from model import *
from controller.controller import *

## Create Ball and Beam Model (real and NN) ----------------------------------------------
real_model = BallAndBeam()
nn_model = BallBeamNN()

## Training ------------------------------------------------------------------------------
loss_history, nn_model = train(nn_model,real_model,N=20000)

## Initial Condition ---------------------------------------------------------------------
x_init = np.array([0.01, 0., 0., 0.])
u_init = 0
xref = np.array([0.2, 0., 0., 0.])

# Simulation -----------------------------------------------------------------------------
x_sim,u_sim,x_nn_sim,u_nn_sim,t_sim = simulate(x_init, u_init, xref, real_model, nn_model, pole_placement, N=50)

# Plot simulation results ----------------------------------------------------------------
plot_sim(x_sim,u_sim,x_nn_sim,u_nn_sim,t_sim)
plot_loss(loss_history)

## Test On Unseen Input ------------------------------------------------------------------
x_test = np.array([0.1, 0.2, 0.05, -0.1])
u_test = 0.3

x_nn = torch.tensor(np.concatenate((x_test, [u_test])), dtype=torch.float32)
x, x_nn = test(x_test,u_test,x_nn,real_model,nn_model,Ts=0.1)

print("\nTrue xdot  :", x)
print("NN Predicted:", x_nn[:-1])