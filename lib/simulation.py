import torch
import numpy as np
import matplotlib.pyplot as plt

def simulate(x,u,xref,real_model,id_model,controller,N=100,Ts=0.1):
    t = [0]
    x_nn = torch.tensor(np.concatenate((x, [u])), dtype=torch.float32)

    x_sim = np.empty((0,4))
    x_nn_sim = np.empty((0,5))
    u_sim = []

    x_sim = np.vstack([x_sim, x])
    x_nn_sim = np.vstack([x_nn_sim, x_nn.numpy()])
    u_sim.append(u)

    for i in range(N):
        # Control
        # u = controller.control(xref[0],x[0])
        u = 0

        xdot_true = real_model.f(x, u)
        x_nn = torch.tensor(np.concatenate((x_nn[:-1], [u])), dtype=torch.float32)
        xdot_pred = id_model(x_nn).detach().numpy()
        
        # Forward Euler method
        x += Ts * xdot_true
        x_nn += Ts * np.concatenate((xdot_pred, [0]))

        x_sim = np.vstack([x_sim, x])
        x_nn_sim = np.vstack([x_nn_sim, x_nn.numpy()])
        u_sim.append(u)

        t.append((i+1)*Ts)

    return x_sim,u_sim,x_nn_sim,t

def plot_sim(x,u,xnn,t):
    fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    ax[0].plot(t, x[:,0], label='Real')
    ax[0].plot(t, xnn[:,0], label='NN')
    ax[1].plot(t, np.rad2deg(x[:,2]), label='Real')
    ax[1].plot(t, np.rad2deg(xnn[:,2]), label='NN')
    
    ax[0].set_ylabel('Ball position (m)')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Beam angle (deg)')
    ax[1].grid(True)
    ax[1].legend()

    plt.suptitle('Ball and Beam - System Identification', fontsize=14, fontweight='bold')
    plt.show(block=False)