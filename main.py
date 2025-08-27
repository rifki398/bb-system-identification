import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from lib import *
from model import *
from controller import *

os.system("cls||clear")

## Create Ball and Beam Model (real and identified)
real_model = BallAndBeam()
id_model = BallBeamNN()

## Dataset Generation
X,Y = generate_dataset(real_model,N=10000)

## Training
loss_history, id_model = train(X,Y,id_model)

## Test On Unseen Input
x_test = np.array([0.1, 0.2, 0.05, -0.1])
u_test = 0.3

x_nn = torch.tensor(np.concatenate((x_test, [u_test])), dtype=torch.float32)
x, x_nn = test(x_test,u_test,x_nn,real_model,id_model,Ts=0.1)



x_init = np.array([0.01, 0., 0., 0.])
u_init = 0
xref = np.array([0.2, 0., 0., 0.])

PID_Controller = PID()

x_sim,u_sim,x_nn_sim,t_sim = simulate(x_init,
                                u_init, 
                                xref,
                                real_model,
                                id_model,
                                PID_Controller,
                                N=50)
plot_sim(x_sim,u_sim,x_nn_sim,t_sim)

print("\nTrue xdot  :", x)
print("NN Predicted:", x_nn[:-1])

## Plot Loss
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.title("Training Loss")
plt.show()
