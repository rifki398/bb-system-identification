import torch
import numpy as np
import matplotlib.pyplot as plt

def simulate(x,u,xref,real_model,id_model,controller,N=100,Ts=0.1):
    t = [0]
    x_nn = torch.tensor(np.concatenate((x, [u])), dtype=torch.float32)

    x_sim = np.empty((0,4))
    x_nn_sim = np.empty((0,5))
    u_sim = np.empty((0,1))
    u_nn_sim = np.empty((0,1))

    x_sim = np.vstack([x_sim, x])
    x_nn_sim = np.vstack([x_nn_sim, x_nn.numpy()])
    u_sim = np.vstack([u_sim, u])
    u_nn_sim = np.vstack([u_nn_sim, u])

    A,B = real_model.linearized_model()
    for i in range(N):
        # Control
        K = controller(A,B)
        u = -K @ (x - xref)
        u_nn = -K @ (x_nn.numpy()[:-1] - xref)

        xdot_true = real_model.f(x, u)
        x_nn = torch.tensor(np.concatenate((x_nn[:-1], u_nn)), dtype=torch.float32)
        xdot_pred = id_model(x_nn).detach().numpy()
        
        # Forward Euler method
        x += Ts * xdot_true
        x_nn += Ts * np.concatenate((xdot_pred, [0]))

        x_sim = np.vstack([x_sim, x])
        x_nn_sim = np.vstack([x_nn_sim, x_nn.numpy()])
        u_sim = np.vstack([u_sim, u])
        u_nn_sim = np.vstack([u_nn_sim, u_nn])

        t.append((i+1)*Ts)

    return x_sim,u_sim,x_nn_sim,u_nn_sim,t

def plot_sim(x,u,xnn,unn,t):
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

    fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    ax[0].plot(t, x[:,1], label='Real')
    ax[0].plot(t, xnn[:,1], label='NN')
    ax[1].plot(t, np.rad2deg(x[:,3]), label='Real')
    ax[1].plot(t, np.rad2deg(xnn[:,3]), label='NN')
    
    ax[0].set_ylabel('Ball velocity (m/s)')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Beam angular velocity (deg/s)')
    ax[1].grid(True)
    ax[1].legend()

    plt.suptitle('Ball and Beam - Velocity', fontsize=14, fontweight='bold')
    plt.show(block=False)

    plt.figure()
    plt.plot(t,u,label="Real model")
    plt.plot(t,unn,label="NN model")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (N/m)")
    plt.grid(True)
    plt.legend()
    plt.title("Control Signal")
    plt.show(block=False)

def plot_loss(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.title("Training Loss")
    plt.show()